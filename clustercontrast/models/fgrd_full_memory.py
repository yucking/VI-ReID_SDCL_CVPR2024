"""Full-Graph Ranking Distillation with far negatives.

This is the training-time counterpart of the Metric Stack diagnostic:

* build the complete train RGB/IR graph, not camera-balanced subgraphs;
* run exact ``re_ranking(base_distance, query_distance, gallery_distance)`` on
  the full combined graph;
* keep top-K reranked neighbours as a soft positive distribution;
* sample negatives only from a farther rank band, so nearby graph neighbours
  are never treated as negatives;
* distill the ordering into the raw/global embedding.
"""

from __future__ import absolute_import

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class FGRDFullRelation(object):
    """Full-graph ranking targets indexed by source train id."""

    candidate_ids: torch.Tensor
    teacher_probs: torch.Tensor
    positive_mask: torch.Tensor
    negative_mask: torch.Tensor
    confidence: torch.Tensor
    valid: torch.Tensor
    positive_count: int
    negative_count: int


def _normalize_torch(features: torch.Tensor) -> torch.Tensor:
    return F.normalize(features, dim=1)


def fuse_metric_features(global_features: torch.Tensor,
                         local_features: torch.Tensor,
                         global_weight: float) -> torch.Tensor:
    if not 0.0 < float(global_weight) < 1.0:
        raise ValueError('global_weight must be in (0, 1).')
    local_weight = 1.0 - float(global_weight)
    return torch.cat((
        math.sqrt(float(global_weight)) * _normalize_torch(global_features),
        math.sqrt(local_weight) * _normalize_torch(local_features),
    ), dim=1)


def _to_numpy(features: torch.Tensor) -> np.ndarray:
    return features.detach().float().cpu().numpy().astype(np.float32, copy=False)


def _normalize_np(features: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.clip(norms, 1e-12, None)


def _cosine_distance(query_features: np.ndarray,
                     gallery_features: np.ndarray) -> np.ndarray:
    return (1.0 - np.matmul(query_features, gallery_features.T)).astype(np.float32)


def _row_minmax(distance: np.ndarray) -> np.ndarray:
    minimum = distance.min(axis=1, keepdims=True)
    maximum = distance.max(axis=1, keepdims=True)
    return ((distance - minimum) /
            np.clip(maximum - minimum, 1e-12, None)).astype(np.float32)


def _csls_distance(query_features: np.ndarray,
                   gallery_features: np.ndarray,
                   neighbors: int) -> np.ndarray:
    similarity = np.matmul(query_features, gallery_features.T).astype(np.float32)
    query_k = min(max(1, int(neighbors)), similarity.shape[1])
    gallery_k = min(max(1, int(neighbors)), similarity.shape[0])
    query_scale = np.partition(
        similarity, similarity.shape[1] - query_k, axis=1
    )[:, -query_k:].mean(axis=1)
    gallery_scale = np.partition(
        similarity.T, similarity.shape[0] - gallery_k, axis=1
    )[:, -gallery_k:].mean(axis=1)
    return (-(2.0 * similarity - query_scale[:, None] -
              gallery_scale[None, :])).astype(np.float32)


def _topk_smallest(distance: np.ndarray, topk: int) -> Tuple[np.ndarray, np.ndarray]:
    k = min(max(1, int(topk)), distance.shape[1])
    if k == distance.shape[1]:
        ids = np.argsort(distance, axis=1)[:, :k]
    else:
        ids = np.argpartition(distance, kth=k - 1, axis=1)[:, :k]
        local = np.take_along_axis(distance, ids, axis=1)
        ids = np.take_along_axis(ids, np.argsort(local, axis=1), axis=1)
    values = np.take_along_axis(distance, ids, axis=1)
    return values.astype(np.float32), ids.astype(np.int64)


def _rank_band(distance: np.ndarray,
               start: int,
               count: int) -> Tuple[np.ndarray, np.ndarray]:
    """Select deterministic negatives from a farther rank band."""
    count = min(max(1, int(count)), distance.shape[1])
    start = min(max(0, int(start)), max(0, distance.shape[1] - count))
    end = min(distance.shape[1], start + count)
    # argpartition to the band end is much cheaper than full argsort on large rows.
    partial = np.argpartition(distance, kth=end - 1, axis=1)[:, :end]
    partial_dist = np.take_along_axis(distance, partial, axis=1)
    order = np.argsort(partial_dist, axis=1)[:, start:end]
    ids = np.take_along_axis(partial, order, axis=1)
    values = np.take_along_axis(distance, ids, axis=1)
    return values.astype(np.float32), ids.astype(np.int64)


def _softmax_negative(distance: np.ndarray, temperature: float) -> np.ndarray:
    logits = -distance / max(float(temperature), 1e-6)
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits).astype(np.float32)
    return exp / np.clip(exp.sum(axis=1, keepdims=True), 1e-12, None)


def _metric_stack_full_cross(rgb_features: np.ndarray,
                             ir_features: np.ndarray,
                             rerank_k1: int,
                             rerank_k2: int,
                             rerank_lambda: float) -> Tuple[np.ndarray, np.ndarray]:
    """Run exact k-reciprocal reranking once on the complete RGB+IR graph."""
    rgb_count = int(rgb_features.shape[0])
    ir_count = int(ir_features.shape[0])
    all_count = rgb_count + ir_count

    rgb_ir = _cosine_distance(rgb_features, ir_features)
    rgb_rgb = _cosine_distance(rgb_features, rgb_features)
    ir_ir = _cosine_distance(ir_features, ir_features)
    original = np.concatenate(
        [np.concatenate([rgb_rgb, rgb_ir], axis=1),
         np.concatenate([rgb_ir.T, ir_ir], axis=1)],
        axis=0,
    )
    del rgb_ir, rgb_rgb, ir_ir

    original = np.power(original, 2).astype(np.float32)
    original = np.transpose(original / np.max(original, axis=0))
    V = np.zeros_like(original, dtype=np.float32)
    initial_rank = np.argsort(original).astype(np.int32)
    k1 = min(int(rerank_k1), max(1, all_count - 1))
    k2 = max(1, int(rerank_k2))

    for i in range(all_count):
        forward = initial_rank[i, :k1 + 1]
        backward = initial_rank[forward, :k1 + 1]
        reciprocal = forward[np.where(backward == i)[0]]
        expansion = reciprocal
        for candidate in reciprocal:
            candidate_forward = initial_rank[
                candidate, :int(np.around(k1 / 2.0)) + 1]
            candidate_backward = initial_rank[
                candidate_forward, :int(np.around(k1 / 2.0)) + 1]
            candidate_reciprocal = candidate_forward[
                np.where(candidate_backward == candidate)[0]]
            if (len(candidate_reciprocal) > 0 and
                    len(np.intersect1d(candidate_reciprocal, reciprocal)) >
                    2.0 / 3.0 * len(candidate_reciprocal)):
                expansion = np.append(expansion, candidate_reciprocal)
        expansion = np.unique(expansion)
        weight = np.exp(-original[i, expansion])
        V[i, expansion] = weight / np.sum(weight)

    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=np.float32)
        for i in range(all_count):
            V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
        V = V_qe
    del initial_rank

    inv_index = [np.where(V[:, i] != 0)[0] for i in range(all_count)]
    rgb_to_ir = np.empty((rgb_count, ir_count), dtype=np.float32)
    ir_to_rgb = np.empty((ir_count, rgb_count), dtype=np.float32)
    for i in range(all_count):
        temp_min = np.zeros((all_count,), dtype=np.float32)
        non_zero = np.where(V[i, :] != 0)[0]
        for nz in non_zero:
            images = inv_index[nz]
            temp_min[images] += np.minimum(V[i, nz], V[images, nz])
        jaccard = 1.0 - temp_min / (2.0 - temp_min)
        final_row = jaccard * (1.0 - float(rerank_lambda)) + original[i] * float(rerank_lambda)
        if i < rgb_count:
            rgb_to_ir[i] = final_row[rgb_count:]
        else:
            ir_to_rgb[i - rgb_count] = final_row[:rgb_count]
    del original, V
    return rgb_to_ir, ir_to_rgb


def _build_relation(distance: np.ndarray,
                    source_count: int,
                    target_count: int,
                    device: torch.device,
                    dtype: torch.dtype,
                    positive_count: int,
                    negative_count: int,
                    negative_start: int,
                    teacher_temperature: float,
                    confidence_floor: float,
                    entropy_ceiling: float) -> Tuple[FGRDFullRelation, Dict[str, float]]:
    positive_count = min(max(1, int(positive_count)), target_count)
    negative_count = min(max(1, int(negative_count)), max(1, target_count - positive_count))
    candidate_count = positive_count + negative_count

    pos_dist, pos_ids = _topk_smallest(distance, positive_count)
    neg_dist, neg_ids = _rank_band(distance, negative_start, negative_count)
    candidate_ids_np = np.concatenate([pos_ids, neg_ids], axis=1)
    probs_np = np.zeros((source_count, candidate_count), dtype=np.float32)
    probs_np[:, :positive_count] = _softmax_negative(pos_dist, teacher_temperature)

    entropy = -(probs_np[:, :positive_count] *
                np.log(np.clip(probs_np[:, :positive_count], 1e-12, None))).sum(axis=1)
    norm_entropy = entropy / math.log(float(max(2, positive_count)))
    margin = neg_dist[:, 0] - pos_dist[:, 0]
    best_score = 1.0 - np.clip(pos_dist[:, 0], 0.0, 1.0)
    row_confidence = (
        0.35 * best_score
        + 0.35 * (1.0 - norm_entropy)
        + 0.30 * np.clip(margin / 0.25, 0.0, 1.0)
    ).astype(np.float32)
    row_valid = (
        (row_confidence >= float(confidence_floor))
        & (norm_entropy <= float(entropy_ceiling))
    )

    positive_mask_np = np.zeros((source_count, candidate_count), dtype=bool)
    negative_mask_np = np.zeros((source_count, candidate_count), dtype=bool)
    positive_mask_np[:, :positive_count] = True
    negative_mask_np[:, positive_count:] = True

    relation = FGRDFullRelation(
        candidate_ids=torch.from_numpy(candidate_ids_np).to(device=device, dtype=torch.long),
        teacher_probs=torch.from_numpy(probs_np).to(device=device, dtype=dtype),
        positive_mask=torch.from_numpy(positive_mask_np).to(device=device),
        negative_mask=torch.from_numpy(negative_mask_np).to(device=device),
        confidence=torch.from_numpy(row_confidence).to(device=device, dtype=dtype),
        valid=torch.from_numpy(row_valid).to(device=device, dtype=torch.bool),
        positive_count=positive_count,
        negative_count=negative_count,
    )
    valid_count = int(row_valid.sum())
    metrics = {
        'coverage': float(valid_count) / float(max(1, source_count)),
        'active_rows': float(valid_count),
        'teacher_entropy': float(norm_entropy[row_valid].mean()) if valid_count else 0.0,
        'mean_confidence': float(row_confidence[row_valid].mean()) if valid_count else 0.0,
        'mean_best_distance': float(pos_dist[:, 0].mean()),
        'mean_pos_far_margin': float(margin.mean()),
        'candidate_count': float(candidate_count),
    }
    return relation, metrics


@torch.no_grad()
def build_fgrd_full_teacher(rgb_features: torch.Tensor,
                            ir_features: torch.Tensor,
                            rgb_view2: Optional[torch.Tensor] = None,
                            ir_view2: Optional[torch.Tensor] = None,
                            rgb_cameras: Optional[torch.Tensor] = None,
                            ir_cameras: Optional[torch.Tensor] = None,
                            global_weight: float = 0.25,
                            query_per_camera: int = -1,
                            gallery_per_camera: int = -1,
                            positive_count: int = 32,
                            negative_count: int = 32,
                            negative_start: int = 256,
                            rerank_k1: int = 30,
                            rerank_k2: int = 3,
                            rerank_lambda: float = 0.10,
                            csls_neighbors: int = 5,
                            csls_blend: float = 0.75,
                            teacher_temperature: float = 0.07,
                            confidence_floor: float = 0.05,
                            entropy_ceiling: float = 0.99,
                            seed: int = 1) -> Tuple[FGRDFullRelation, FGRDFullRelation, Dict[str, float]]:
    del rgb_cameras, ir_cameras, query_per_camera, gallery_per_camera, seed
    if rgb_view2 is None:
        rgb_view2 = rgb_features
    if ir_view2 is None:
        ir_view2 = ir_features

    rgb_fused = fuse_metric_features(rgb_features, rgb_view2, global_weight)
    ir_fused = fuse_metric_features(ir_features, ir_view2, global_weight)
    rgb_np = _normalize_np(_to_numpy(rgb_fused))
    ir_np = _normalize_np(_to_numpy(ir_fused))

    rerank_rgb_ir, rerank_ir_rgb = _metric_stack_full_cross(
        rgb_np, ir_np, rerank_k1, rerank_k2, rerank_lambda)
    csls_rgb_ir = _csls_distance(rgb_np, ir_np, csls_neighbors)
    final_rgb_ir = (
        float(csls_blend) * _row_minmax(rerank_rgb_ir)
        + (1.0 - float(csls_blend)) * _row_minmax(csls_rgb_ir)
    ).astype(np.float32, copy=False)
    final_ir_rgb = (
        float(csls_blend) * _row_minmax(rerank_ir_rgb)
        + (1.0 - float(csls_blend)) * _row_minmax(csls_rgb_ir.T)
    ).astype(np.float32, copy=False)
    del rerank_rgb_ir, rerank_ir_rgb, csls_rgb_ir

    rgb_to_ir, rgb_stats = _build_relation(
        final_rgb_ir, int(rgb_features.size(0)), int(ir_features.size(0)),
        rgb_features.device, rgb_features.dtype, positive_count, negative_count,
        negative_start, teacher_temperature, confidence_floor, entropy_ceiling)
    ir_to_rgb, ir_stats = _build_relation(
        final_ir_rgb, int(ir_features.size(0)), int(rgb_features.size(0)),
        ir_features.device, ir_features.dtype, positive_count, negative_count,
        negative_start, teacher_temperature, confidence_floor, entropy_ceiling)

    metrics = {
        'coverage_rgb': rgb_stats['coverage'],
        'coverage_ir': ir_stats['coverage'],
        'query_rgb': float(rgb_features.size(0)),
        'query_ir': float(ir_features.size(0)),
        'gallery_ir': float(ir_features.size(0)),
        'gallery_rgb': float(rgb_features.size(0)),
        'exact_nodes_rgb_to_ir': float(rgb_features.size(0) + ir_features.size(0)),
        'exact_nodes_ir_to_rgb': float(rgb_features.size(0) + ir_features.size(0)),
        'active_rows_rgb': rgb_stats['active_rows'],
        'active_rows_ir': ir_stats['active_rows'],
        'candidate_count': max(rgb_stats['candidate_count'], ir_stats['candidate_count']),
        'teacher_entropy': 0.5 * (rgb_stats['teacher_entropy'] + ir_stats['teacher_entropy']),
        'mean_confidence': 0.5 * (rgb_stats['mean_confidence'] + ir_stats['mean_confidence']),
        'mean_best_distance': 0.5 * (rgb_stats['mean_best_distance'] + ir_stats['mean_best_distance']),
        'mean_pos_far_margin': 0.5 * (rgb_stats['mean_pos_far_margin'] + ir_stats['mean_pos_far_margin']),
    }
    return rgb_to_ir, ir_to_rgb, metrics


def fgrd_full_ranking_loss(student_features: torch.Tensor,
                           source_indices: torch.Tensor,
                           target_memory: torch.Tensor,
                           relation: Optional[FGRDFullRelation],
                           temperature: float = 0.05,
                           margin: float = 0.20,
                           pairwise_weight: float = 0.50) -> Tuple[torch.Tensor, Dict[str, float]]:
    zero = student_features.sum() * 0.0
    if relation is None or source_indices.numel() == 0:
        return zero, {'active_rows': 0.0, 'mean_confidence': 0.0,
                      'listwise': 0.0, 'pairwise': 0.0}

    source_indices = source_indices.long().view(-1)
    if source_indices.device != relation.candidate_ids.device:
        source_indices = source_indices.to(relation.candidate_ids.device)
    candidate_ids = relation.candidate_ids.index_select(0, source_indices)
    teacher_probs = relation.teacher_probs.index_select(0, source_indices)
    positive_mask = relation.positive_mask.index_select(0, source_indices)
    negative_mask = relation.negative_mask.index_select(0, source_indices)
    confidence = relation.confidence.index_select(0, source_indices)
    valid = relation.valid.index_select(0, source_indices)

    row_valid = valid & positive_mask.any(dim=1) & negative_mask.any(dim=1)
    if not bool(row_valid.any().item()):
        return zero, {'active_rows': 0.0, 'mean_confidence': 0.0,
                      'listwise': 0.0, 'pairwise': 0.0}

    target_memory = F.normalize(target_memory.detach(), dim=1)
    candidate_features = target_memory[candidate_ids.clamp_min(0)]
    student = F.normalize(student_features, dim=1)
    logits = torch.bmm(candidate_features, student.unsqueeze(2)).squeeze(2)
    logits = logits / max(float(temperature), 1e-6)

    target = teacher_probs * positive_mask.to(teacher_probs.dtype)
    target = target / target.sum(dim=1, keepdim=True).clamp_min(1e-12)
    listwise = -(target * F.log_softmax(logits, dim=1)).sum(dim=1)

    pos_logits = logits.masked_fill(~positive_mask, -10000.0)
    neg_logits = logits.masked_fill(~negative_mask, -10000.0)
    pair_values = F.relu(float(margin) + neg_logits.unsqueeze(1) - pos_logits.unsqueeze(2))
    pair_mask = positive_mask.unsqueeze(2) & negative_mask.unsqueeze(1)
    pairwise = (
        pair_values * pair_mask.to(pair_values.dtype)
    ).sum(dim=(1, 2)) / pair_mask.sum(dim=(1, 2)).clamp_min(1).to(pair_values.dtype)

    weights = confidence * row_valid.to(confidence.dtype)
    per_row = listwise + float(pairwise_weight) * pairwise
    loss = (per_row * weights).sum() / weights.sum().clamp_min(1e-12)
    return loss, {
        'active_rows': float(row_valid.sum().item()),
        'mean_confidence': float(weights[row_valid].mean().item()),
        'listwise': float(listwise[row_valid].mean().item()),
        'pairwise': float(pairwise[row_valid].mean().item()),
    }
