"""Full-Graph Ranking Distillation (FGRD).

FGRD is the training-time version of the Metric Stack signal that was strong
at test time.  It builds an exact ``re_ranking(base_distance, query_distance,
gallery_distance)`` graph on deterministic camera-balanced train subsets, then
distills that graph into raw cosine embeddings with a listwise positive-vs-hard
negative ranking loss.

Unlike the earlier sparse-edge attempt, this module does not reduce the teacher
to a few pull-only edges: each selected source row keeps mutual high-confidence
positives and hard opposite-modality negatives from the same reranked row.  The
trainer then asks the raw/global embedding to rank the positives above those
hard negatives.
"""

from __future__ import absolute_import

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from clustercontrast.utils.rerank import re_ranking


@dataclass
class FGRDRelation(object):
    """Full-rerank ranking targets indexed by source train id."""

    candidate_ids: torch.Tensor
    teacher_probs: torch.Tensor
    positive_mask: torch.Tensor
    negative_mask: torch.Tensor
    confidence: torch.Tensor
    valid: torch.Tensor
    positive_count: int
    hard_negative_count: int


def _normalize_torch(features: torch.Tensor) -> torch.Tensor:
    return F.normalize(features, dim=1)


def fuse_metric_features(global_features: torch.Tensor,
                         local_features: torch.Tensor,
                         global_weight: float) -> torch.Tensor:
    """Match the fixed test-time Metric Stack global/local fusion."""
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
    """Same CSLS distance used by the fixed Metric Stack evaluation."""
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


def _camera_balanced_indices(cameras: Optional[torch.Tensor],
                             total: int,
                             per_camera: int,
                             seed: int) -> np.ndarray:
    """Deterministic camera-balanced subset; ``per_camera <= 0`` means all."""
    if total <= 0:
        return np.empty((0,), dtype=np.int64)
    if cameras is None:
        count = total if int(per_camera) <= 0 else min(total, int(per_camera))
        return np.arange(count, dtype=np.int64)

    cams = np.asarray(cameras.detach().cpu().numpy(), dtype=np.int64).reshape(-1)
    if cams.shape[0] != int(total):
        raise ValueError('camera array length does not match feature count.')
    if int(per_camera) <= 0:
        return np.arange(total, dtype=np.int64)

    rng = np.random.default_rng(int(seed))
    selected = []
    for cam in sorted(np.unique(cams).tolist()):
        ids = np.flatnonzero(cams == cam).astype(np.int64)
        if ids.size > int(per_camera):
            ids = rng.choice(ids, size=int(per_camera), replace=False)
        selected.append(np.sort(ids))
    if not selected:
        return np.empty((0,), dtype=np.int64)
    return np.sort(np.concatenate(selected).astype(np.int64))


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


def _softmax_negative(distance: np.ndarray, temperature: float) -> np.ndarray:
    logits = -distance / max(float(temperature), 1e-6)
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits).astype(np.float32)
    return exp / np.clip(exp.sum(axis=1, keepdims=True), 1e-12, None)


def _mutual_positive_mask(distance: np.ndarray,
                          selected_pos: np.ndarray,
                          positive_count: int,
                          mutual_topk: int) -> np.ndarray:
    """Whether each top positive also sees the source row in its reverse top-k."""
    query_count = int(distance.shape[0])
    if positive_count <= 0:
        return np.zeros((query_count, 0), dtype=bool)
    if int(mutual_topk) <= 0:
        return np.ones((query_count, positive_count), dtype=bool)

    reverse_k = min(max(1, int(mutual_topk)), query_count)
    reverse_top = np.argpartition(distance, kth=reverse_k - 1, axis=0)[:reverse_k, :]
    mask = np.zeros((query_count, positive_count), dtype=bool)
    for row in range(query_count):
        for col in range(positive_count):
            target_pos = int(selected_pos[row, col])
            mask[row, col] = bool(np.any(reverse_top[:, target_pos] == row))
    return mask


@torch.no_grad()
def _build_direction(source_fused: torch.Tensor,
                     target_fused: torch.Tensor,
                     source_cameras: Optional[torch.Tensor],
                     target_cameras: Optional[torch.Tensor],
                     query_per_camera: int,
                     gallery_per_camera: int,
                     positive_count: int,
                     hard_negative_count: int,
                     mutual_topk: int,
                     rerank_k1: int,
                     rerank_k2: int,
                     rerank_lambda: float,
                     csls_neighbors: int,
                     csls_blend: float,
                     teacher_temperature: float,
                     confidence_floor: float,
                     entropy_ceiling: float,
                     seed: int) -> Tuple[FGRDRelation, Dict[str, float]]:
    """Build one exact Metric Stack source -> target ranking teacher."""
    if source_fused.dim() != 2 or target_fused.dim() != 2:
        raise ValueError('FGRD expects [num_samples, feature_dim] tensors.')
    if source_fused.size(1) != target_fused.size(1):
        raise ValueError('source and target feature dims differ.')

    device = source_fused.device
    dtype = source_fused.dtype
    source_count = int(source_fused.size(0))
    target_count = int(target_fused.size(0))
    positive_count = min(max(1, int(positive_count)), max(1, target_count))
    hard_negative_count = min(max(1, int(hard_negative_count)),
                              max(1, target_count - positive_count))
    candidate_count = min(target_count, positive_count + hard_negative_count)

    candidate_ids = torch.full((source_count, candidate_count), -1,
                               dtype=torch.long, device=device)
    teacher_probs = torch.zeros((source_count, candidate_count),
                                dtype=dtype, device=device)
    positive_mask = torch.zeros((source_count, candidate_count),
                                dtype=torch.bool, device=device)
    negative_mask = torch.zeros((source_count, candidate_count),
                                dtype=torch.bool, device=device)
    confidence = torch.zeros(source_count, dtype=dtype, device=device)
    valid = torch.zeros(source_count, dtype=torch.bool, device=device)

    query_ids = _camera_balanced_indices(
        source_cameras, source_count, query_per_camera, seed)
    gallery_ids = _camera_balanced_indices(
        target_cameras, target_count, gallery_per_camera, seed + 7919)
    if query_ids.size == 0 or gallery_ids.size == 0:
        relation = FGRDRelation(
            candidate_ids=candidate_ids.detach(),
            teacher_probs=teacher_probs.detach(),
            positive_mask=positive_mask.detach(),
            negative_mask=negative_mask.detach(),
            confidence=confidence.detach(),
            valid=valid.detach(),
            positive_count=positive_count,
            hard_negative_count=hard_negative_count,
        )
        return relation, {
            'coverage': 0.0,
            'query_count': float(query_ids.size),
            'gallery_count': float(gallery_ids.size),
            'teacher_entropy': 0.0,
            'active_rows': 0.0,
            'mean_confidence': 0.0,
            'mean_best_distance': 0.0,
            'mean_pos_neg_margin': 0.0,
            'mutual_positive_rate': 0.0,
            'exact_nodes': float(query_ids.size + gallery_ids.size),
            'candidate_count': float(candidate_count),
        }

    source_np = _normalize_np(_to_numpy(source_fused)[query_ids])
    target_np = _normalize_np(_to_numpy(target_fused)[gallery_ids])
    base_distance = _cosine_distance(source_np, target_np)
    query_distance = _cosine_distance(source_np, source_np)
    gallery_distance = _cosine_distance(target_np, target_np)

    exact_distance = re_ranking(
        base_distance,
        query_distance,
        gallery_distance,
        k1=min(int(rerank_k1), max(1, query_ids.size + gallery_ids.size - 1)),
        k2=max(1, int(rerank_k2)),
        lambda_value=float(rerank_lambda),
    ).astype(np.float32, copy=False)
    csls = _csls_distance(source_np, target_np, int(csls_neighbors))
    final_distance = (
        float(csls_blend) * _row_minmax(exact_distance)
        + (1.0 - float(csls_blend)) * _row_minmax(csls)
    ).astype(np.float32, copy=False)

    selected_dist, selected_pos = _topk_smallest(final_distance, candidate_count)
    selected_target_ids = gallery_ids[selected_pos]
    mutual_mask = _mutual_positive_mask(
        final_distance, selected_pos[:, :positive_count],
        positive_count, mutual_topk)
    hard_negative_mask = np.zeros((query_ids.size, candidate_count), dtype=bool)
    hard_negative_mask[:, positive_count:candidate_count] = True

    positive_dist = selected_dist[:, :positive_count]
    positive_probs = _softmax_negative(positive_dist, teacher_temperature)
    entropy = -(positive_probs * np.log(np.clip(positive_probs, 1e-12, None))).sum(axis=1)
    norm_entropy = entropy / math.log(float(max(2, positive_count)))

    if candidate_count > positive_count:
        margin = selected_dist[:, positive_count] - selected_dist[:, 0]
    else:
        margin = np.ones((selected_dist.shape[0],), dtype=np.float32)
    best_score = 1.0 - np.clip(selected_dist[:, 0], 0.0, 1.0)
    mutual_count = mutual_mask.sum(axis=1).astype(np.float32)
    row_confidence = (
        0.30 * best_score
        + 0.30 * (1.0 - norm_entropy)
        + 0.25 * np.clip(margin / 0.15, 0.0, 1.0)
        + 0.15 * np.clip(mutual_count / float(max(1, positive_count)), 0.0, 1.0)
    ).astype(np.float32)
    row_valid = (
        (mutual_count >= 1.0)
        & (row_confidence >= float(confidence_floor))
        & (norm_entropy <= float(entropy_ceiling))
    )

    row_teacher = np.zeros((query_ids.size, candidate_count), dtype=np.float32)
    row_teacher[:, :positive_count] = positive_probs * mutual_mask.astype(np.float32)
    row_teacher = row_teacher / np.clip(row_teacher.sum(axis=1, keepdims=True), 1e-12, None)

    q_tensor = torch.from_numpy(query_ids).to(device=device, dtype=torch.long)
    candidate_ids[q_tensor] = torch.from_numpy(selected_target_ids).to(
        device=device, dtype=torch.long)
    teacher_probs[q_tensor] = torch.from_numpy(row_teacher).to(device=device, dtype=dtype)

    pos_mask_np = np.zeros((query_ids.size, candidate_count), dtype=bool)
    pos_mask_np[:, :positive_count] = mutual_mask
    positive_mask[q_tensor] = torch.from_numpy(pos_mask_np).to(device=device)
    negative_mask[q_tensor] = torch.from_numpy(hard_negative_mask).to(device=device)
    confidence[q_tensor] = torch.from_numpy(row_confidence).to(device=device, dtype=dtype)
    valid[q_tensor] = torch.from_numpy(row_valid).to(device=device, dtype=torch.bool)

    valid_count = int(row_valid.sum())
    relation = FGRDRelation(
        candidate_ids=candidate_ids.detach(),
        teacher_probs=teacher_probs.detach(),
        positive_mask=positive_mask.detach(),
        negative_mask=negative_mask.detach(),
        confidence=confidence.detach(),
        valid=valid.detach(),
        positive_count=positive_count,
        hard_negative_count=hard_negative_count,
    )
    metrics = {
        'coverage': float(valid_count) / float(max(1, source_count)),
        'query_count': float(query_ids.size),
        'gallery_count': float(gallery_ids.size),
        'teacher_entropy': float(norm_entropy[row_valid].mean())
        if valid_count > 0 else 0.0,
        'active_rows': float(valid_count),
        'mean_confidence': float(row_confidence[row_valid].mean())
        if valid_count > 0 else 0.0,
        'mean_best_distance': float(selected_dist[:, 0].mean()),
        'mean_pos_neg_margin': float(margin.mean()),
        'mutual_positive_rate': float(mutual_mask.mean()) if mutual_mask.size else 0.0,
        'exact_nodes': float(query_ids.size + gallery_ids.size),
        'candidate_count': float(candidate_count),
    }
    return relation, metrics


@torch.no_grad()
def build_fgrd_teacher(rgb_features: torch.Tensor,
                       ir_features: torch.Tensor,
                       rgb_view2: Optional[torch.Tensor] = None,
                       ir_view2: Optional[torch.Tensor] = None,
                       rgb_cameras: Optional[torch.Tensor] = None,
                       ir_cameras: Optional[torch.Tensor] = None,
                       global_weight: float = 0.25,
                       query_per_camera: int = 1536,
                       gallery_per_camera: int = 3072,
                       positive_count: int = 4,
                       hard_negative_count: int = 28,
                       mutual_topk: int = 24,
                       rerank_k1: int = 30,
                       rerank_k2: int = 3,
                       rerank_lambda: float = 0.10,
                       csls_neighbors: int = 5,
                       csls_blend: float = 0.75,
                       teacher_temperature: float = 0.07,
                       confidence_floor: float = 0.20,
                       entropy_ceiling: float = 0.95,
                       seed: int = 1) -> Tuple[FGRDRelation, FGRDRelation, Dict[str, float]]:
    """Build RGB->IR and IR->RGB full-rerank ranking teachers."""
    if rgb_view2 is None:
        rgb_view2 = rgb_features
    if ir_view2 is None:
        ir_view2 = ir_features

    rgb_fused = fuse_metric_features(rgb_features, rgb_view2, global_weight)
    ir_fused = fuse_metric_features(ir_features, ir_view2, global_weight)

    rgb_to_ir, rgb_stats = _build_direction(
        rgb_fused, ir_fused, rgb_cameras, ir_cameras,
        query_per_camera, gallery_per_camera, positive_count,
        hard_negative_count, mutual_topk, rerank_k1, rerank_k2,
        rerank_lambda, csls_neighbors, csls_blend, teacher_temperature,
        confidence_floor, entropy_ceiling, seed)
    ir_to_rgb, ir_stats = _build_direction(
        ir_fused, rgb_fused, ir_cameras, rgb_cameras,
        query_per_camera, gallery_per_camera, positive_count,
        hard_negative_count, mutual_topk, rerank_k1, rerank_k2,
        rerank_lambda, csls_neighbors, csls_blend, teacher_temperature,
        confidence_floor, entropy_ceiling, seed + 104729)

    metrics = {
        'coverage_rgb': rgb_stats['coverage'],
        'coverage_ir': ir_stats['coverage'],
        'query_rgb': rgb_stats['query_count'],
        'query_ir': ir_stats['query_count'],
        'gallery_ir': rgb_stats['gallery_count'],
        'gallery_rgb': ir_stats['gallery_count'],
        'exact_nodes_rgb_to_ir': rgb_stats['exact_nodes'],
        'exact_nodes_ir_to_rgb': ir_stats['exact_nodes'],
        'active_rows_rgb': rgb_stats['active_rows'],
        'active_rows_ir': ir_stats['active_rows'],
        'candidate_count': max(rgb_stats['candidate_count'], ir_stats['candidate_count']),
        'teacher_entropy': 0.5 * (
            rgb_stats['teacher_entropy'] + ir_stats['teacher_entropy']),
        'mean_confidence': 0.5 * (
            rgb_stats['mean_confidence'] + ir_stats['mean_confidence']),
        'mean_best_distance': 0.5 * (
            rgb_stats['mean_best_distance'] + ir_stats['mean_best_distance']),
        'mean_pos_neg_margin': 0.5 * (
            rgb_stats['mean_pos_neg_margin'] + ir_stats['mean_pos_neg_margin']),
        'mutual_positive_rate': 0.5 * (
            rgb_stats['mutual_positive_rate'] + ir_stats['mutual_positive_rate']),
    }
    return rgb_to_ir, ir_to_rgb, metrics


def fgrd_ranking_loss(student_features: torch.Tensor,
                      source_indices: torch.Tensor,
                      target_memory: torch.Tensor,
                      relation: Optional[FGRDRelation],
                      temperature: float = 0.05,
                      margin: float = 0.15,
                      pairwise_weight: float = 0.50) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Distill full-rerank positive/hard-negative ordering into raw cosine."""
    zero = student_features.sum() * 0.0
    if relation is None or source_indices.numel() == 0:
        return zero, {
            'active_rows': 0.0,
            'mean_confidence': 0.0,
            'listwise': 0.0,
            'pairwise': 0.0,
        }

    source_indices = source_indices.long().view(-1)
    if source_indices.device != relation.candidate_ids.device:
        source_indices = source_indices.to(relation.candidate_ids.device)
    candidate_ids = relation.candidate_ids.index_select(0, source_indices)
    teacher_probs = relation.teacher_probs.index_select(0, source_indices)
    positive_mask = relation.positive_mask.index_select(0, source_indices)
    negative_mask = relation.negative_mask.index_select(0, source_indices)
    confidence = relation.confidence.index_select(0, source_indices)
    valid = relation.valid.index_select(0, source_indices)

    candidate_valid = candidate_ids.ge(0)
    positive_mask = positive_mask & candidate_valid
    negative_mask = negative_mask & candidate_valid
    row_valid = valid & positive_mask.any(dim=1) & negative_mask.any(dim=1)
    if not bool(row_valid.any().item()):
        return zero, {
            'active_rows': 0.0,
            'mean_confidence': 0.0,
            'listwise': 0.0,
            'pairwise': 0.0,
        }

    target_memory = F.normalize(target_memory.detach(), dim=1)
    safe_ids = candidate_ids.clamp_min(0)
    candidate_features = target_memory[safe_ids]
    student = F.normalize(student_features, dim=1)
    logits = torch.bmm(candidate_features, student.unsqueeze(2)).squeeze(2)
    logits = logits / max(float(temperature), 1e-6)
    logits = logits.masked_fill(~candidate_valid, -10000.0)

    target = teacher_probs * positive_mask.to(teacher_probs.dtype)
    target = target / target.sum(dim=1, keepdim=True).clamp_min(1e-12)
    log_probs = F.log_softmax(logits, dim=1)
    listwise = -(target * log_probs).sum(dim=1)

    pos_logits = logits.masked_fill(~positive_mask, -10000.0)
    neg_logits = logits.masked_fill(~negative_mask, -10000.0)
    pair_values = F.relu(
        float(margin) + neg_logits.unsqueeze(1) - pos_logits.unsqueeze(2))
    pair_mask = positive_mask.unsqueeze(2) & negative_mask.unsqueeze(1)
    pairwise = (
        pair_values * pair_mask.to(pair_values.dtype)
    ).sum(dim=(1, 2)) / pair_mask.sum(dim=(1, 2)).clamp_min(1).to(pair_values.dtype)

    weights = confidence * row_valid.to(confidence.dtype)
    per_row = listwise + float(pairwise_weight) * pairwise
    loss = (per_row * weights).sum() / weights.sum().clamp_min(1e-12)
    active = row_valid.sum().item()
    return loss, {
        'active_rows': float(active),
        'mean_confidence': float(weights[row_valid].mean().item()),
        'listwise': float(listwise[row_valid].mean().item()),
        'pairwise': float(pairwise[row_valid].mean().item()),
    }
