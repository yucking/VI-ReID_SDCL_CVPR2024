"""Exact Metric-Stack Edge Distillation (EMSED).

The previous MSRD teacher only approximated the Metric Stack graph.  EMSED
keeps the part that actually moved the SYSU numbers in the frozen-checkpoint
tests: global/local fusion, full k-reciprocal ``re_ranking(base, query,
gallery)``, and the final CSLS blend.  To keep this usable during training, the
exact graph is built on deterministic camera-balanced train subsets and only
the strongest opposite-modality edges are distilled into the raw embedding.
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
class EMSEDRelation(object):
    """Sparse exact-rerank cross-modal edges indexed by source train id."""

    candidate_ids: torch.Tensor
    teacher_probs: torch.Tensor
    confidence: torch.Tensor
    valid: torch.Tensor
    stable_count: torch.Tensor
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
    """Same CSLS distance used by ``test_sysu_metricstack.py``."""
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


@torch.no_grad()
def _build_direction(source_fused: torch.Tensor,
                     target_fused: torch.Tensor,
                     source_cameras: Optional[torch.Tensor],
                     target_cameras: Optional[torch.Tensor],
                     query_per_camera: int,
                     gallery_per_camera: int,
                     edge_topk: int,
                     rerank_k1: int,
                     rerank_k2: int,
                     rerank_lambda: float,
                     csls_neighbors: int,
                     csls_blend: float,
                     teacher_temperature: float,
                     confidence_floor: float,
                     seed: int) -> Tuple[EMSEDRelation, Dict[str, float]]:
    """Build one exact Metric-Stack source -> target edge teacher."""
    if source_fused.dim() != 2 or target_fused.dim() != 2:
        raise ValueError('EMSED expects [num_samples, feature_dim] tensors.')
    if source_fused.size(1) != target_fused.size(1):
        raise ValueError('source and target feature dims differ.')

    device = source_fused.device
    source_count = int(source_fused.size(0))
    target_count = int(target_fused.size(0))
    count = min(max(1, int(edge_topk)), max(1, target_count))

    candidate_ids = torch.full((source_count, count), -1, dtype=torch.long,
                               device=device)
    teacher_probs = torch.zeros((source_count, count), dtype=source_fused.dtype,
                                device=device)
    confidence = torch.zeros(source_count, dtype=source_fused.dtype,
                             device=device)
    valid = torch.zeros(source_count, dtype=torch.bool, device=device)
    stable_count = torch.zeros(source_count, dtype=torch.long, device=device)

    query_ids = _camera_balanced_indices(
        source_cameras, source_count, query_per_camera, seed)
    gallery_ids = _camera_balanced_indices(
        target_cameras, target_count, gallery_per_camera, seed + 7919)
    if query_ids.size == 0 or gallery_ids.size == 0:
        relation = EMSEDRelation(
            candidate_ids=candidate_ids.detach(),
            teacher_probs=teacher_probs.detach(),
            confidence=confidence.detach(),
            valid=valid.detach(),
            stable_count=stable_count.detach(),
            positive_count=count,
            hard_negative_count=0,
        )
        return relation, {
            'coverage': 0.0,
            'query_count': float(query_ids.size),
            'gallery_count': float(gallery_ids.size),
            'teacher_entropy': 0.0,
            'active_rows': 0.0,
            'mean_confidence': 0.0,
            'mean_best_distance': 0.0,
            'mean_margin': 0.0,
            'exact_nodes': float(query_ids.size + gallery_ids.size),
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

    selected_dist, selected_pos = _topk_smallest(final_distance, count)
    selected_target_ids = gallery_ids[selected_pos]
    probabilities = _softmax_negative(selected_dist, teacher_temperature)

    if count > 1:
        margin = selected_dist[:, 1] - selected_dist[:, 0]
    else:
        margin = np.ones((selected_dist.shape[0],), dtype=np.float32)
    entropy = -(probabilities * np.log(np.clip(probabilities, 1e-12, None))).sum(axis=1)
    norm_entropy = entropy / math.log(float(max(2, count)))
    best_score = 1.0 - np.clip(selected_dist[:, 0], 0.0, 1.0)
    row_confidence = (
        0.40 * best_score
        + 0.40 * (1.0 - norm_entropy)
        + 0.20 * np.clip(margin / 0.10, 0.0, 1.0)
    ).astype(np.float32)
    row_valid = row_confidence >= float(confidence_floor)
    stable = (selected_dist < selected_dist[:, :1] + 0.10).sum(axis=1)

    q_tensor = torch.from_numpy(query_ids).to(device=device, dtype=torch.long)
    candidate_ids[q_tensor] = torch.from_numpy(selected_target_ids).to(
        device=device, dtype=torch.long)
    teacher_probs[q_tensor] = torch.from_numpy(probabilities).to(
        device=device, dtype=source_fused.dtype)
    confidence[q_tensor] = torch.from_numpy(row_confidence).to(
        device=device, dtype=source_fused.dtype)
    valid[q_tensor] = torch.from_numpy(row_valid).to(device=device, dtype=torch.bool)
    stable_count[q_tensor] = torch.from_numpy(stable.astype(np.int64)).to(
        device=device, dtype=torch.long)

    valid_count = int(row_valid.sum())
    relation = EMSEDRelation(
        candidate_ids=candidate_ids.detach(),
        teacher_probs=teacher_probs.detach(),
        confidence=confidence.detach(),
        valid=valid.detach(),
        stable_count=stable_count.detach(),
        positive_count=count,
        hard_negative_count=0,
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
        'mean_margin': float(margin.mean()),
        'exact_nodes': float(query_ids.size + gallery_ids.size),
    }
    return relation, metrics


@torch.no_grad()
def build_emsed_teacher(rgb_features: torch.Tensor,
                        ir_features: torch.Tensor,
                        rgb_view2: Optional[torch.Tensor] = None,
                        ir_view2: Optional[torch.Tensor] = None,
                        rgb_cameras: Optional[torch.Tensor] = None,
                        ir_cameras: Optional[torch.Tensor] = None,
                        global_weight: float = 0.25,
                        query_per_camera: int = 1024,
                        gallery_per_camera: int = 2048,
                        edge_topk: int = 3,
                        rerank_k1: int = 30,
                        rerank_k2: int = 3,
                        rerank_lambda: float = 0.10,
                        csls_neighbors: int = 5,
                        csls_blend: float = 0.75,
                        teacher_temperature: float = 0.07,
                        confidence_floor: float = 0.0,
                        seed: int = 1) -> Tuple[EMSEDRelation, EMSEDRelation, Dict[str, float]]:
    """Build RGB->IR and IR->RGB exact Metric-Stack edge teachers."""
    if rgb_view2 is None:
        rgb_view2 = rgb_features
    if ir_view2 is None:
        ir_view2 = ir_features

    rgb_fused = fuse_metric_features(rgb_features, rgb_view2, global_weight)
    ir_fused = fuse_metric_features(ir_features, ir_view2, global_weight)

    rgb_to_ir, rgb_stats = _build_direction(
        rgb_fused, ir_fused, rgb_cameras, ir_cameras,
        query_per_camera, gallery_per_camera, edge_topk,
        rerank_k1, rerank_k2, rerank_lambda, csls_neighbors, csls_blend,
        teacher_temperature, confidence_floor, seed)
    ir_to_rgb, ir_stats = _build_direction(
        ir_fused, rgb_fused, ir_cameras, rgb_cameras,
        query_per_camera, gallery_per_camera, edge_topk,
        rerank_k1, rerank_k2, rerank_lambda, csls_neighbors, csls_blend,
        teacher_temperature, confidence_floor, seed + 104729)

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
        'teacher_entropy': 0.5 * (
            rgb_stats['teacher_entropy'] + ir_stats['teacher_entropy']),
        'mean_confidence': 0.5 * (
            rgb_stats['mean_confidence'] + ir_stats['mean_confidence']),
        'mean_best_distance': 0.5 * (
            rgb_stats['mean_best_distance'] + ir_stats['mean_best_distance']),
        'mean_margin': 0.5 * (
            rgb_stats['mean_margin'] + ir_stats['mean_margin']),
    }
    return rgb_to_ir, ir_to_rgb, metrics


def emsed_edge_loss(student_features: torch.Tensor,
                    source_indices: torch.Tensor,
                    target_memory: torch.Tensor,
                    relation: Optional[EMSEDRelation],
                    temperature: float = 0.05) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Pull raw embeddings toward exact-rerank positive edges only.

    ``temperature`` is kept in the signature for trainer compatibility and
    future margin extensions; the current loss is a bounded cosine pull rather
    than the noisy many-candidate KL used in v10.
    """
    del temperature
    zero = student_features.sum() * 0.0
    if relation is None or source_indices.numel() == 0:
        return zero, {'active_rows': 0.0, 'mean_confidence': 0.0}

    source_indices = source_indices.long().view(-1)
    if source_indices.device != relation.candidate_ids.device:
        source_indices = source_indices.to(relation.candidate_ids.device)
    candidate_ids = relation.candidate_ids.index_select(0, source_indices)
    teacher_probs = relation.teacher_probs.index_select(0, source_indices)
    confidence = relation.confidence.index_select(0, source_indices)
    valid = relation.valid.index_select(0, source_indices)
    candidate_valid = candidate_ids.ge(0)
    row_valid = valid & candidate_valid.any(dim=1)
    if not bool(row_valid.any().item()):
        return zero, {'active_rows': 0.0, 'mean_confidence': 0.0}

    target_memory = F.normalize(target_memory.detach(), dim=1)
    safe_ids = candidate_ids.clamp_min(0)
    candidate_features = target_memory[safe_ids]
    student = F.normalize(student_features, dim=1)
    similarity = torch.bmm(candidate_features, student.unsqueeze(2)).squeeze(2)
    target = teacher_probs * candidate_valid.to(teacher_probs.dtype)
    target = target / target.sum(dim=1, keepdim=True).clamp_min(1e-12)

    pull = (target * (1.0 - similarity)).sum(dim=1)
    weights = confidence * row_valid.to(confidence.dtype)
    loss = (pull * weights).sum() / weights.sum().clamp_min(1e-12)
    return loss, {
        'active_rows': float(row_valid.sum().item()),
        'mean_confidence': float(weights[row_valid].mean().item()),
    }
