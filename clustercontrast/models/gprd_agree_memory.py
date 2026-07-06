"""Raw-agreement Graph-Preserved Reciprocal Distillation (GPRD-AGREE).

GPRD is the training-time version of the Metric Stack signal that was strong
at test time.  It builds an exact ``re_ranking(base_distance, query_distance,
gallery_distance)`` graph on deterministic camera-balanced train subsets, then
distills that graph into raw cosine embeddings with a soft neighbourhood
distribution loss.

Unlike GPRD-SOFT, this variant keeps the soft neighbourhood but records whether
each Metric Stack edge is also plausible in the current raw embedding.  That
raw-agreement signal gates row confidence and the auxiliary loss so the teacher
does not over-train edges that the student geometry strongly contradicts.
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
class GPRDRelation(object):
    """stable reciprocal ranking targets indexed by source train id."""

    candidate_ids: torch.Tensor
    teacher_probs: torch.Tensor
    positive_mask: torch.Tensor
    negative_mask: torch.Tensor
    raw_agreement: torch.Tensor
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


def _camera_balance_entropy(cameras: Optional[torch.Tensor],
                            selected_ids: np.ndarray) -> float:
    if cameras is None or selected_ids.size == 0:
        return 1.0 if selected_ids.size > 0 else 0.0
    cams = np.asarray(cameras.detach().cpu().numpy(), dtype=np.int64).reshape(-1)
    selected_cams = cams[selected_ids]
    _, counts = np.unique(selected_cams, return_counts=True)
    if counts.size <= 1:
        return 0.0
    probs = counts.astype(np.float32) / float(counts.sum())
    entropy = -(probs * np.log(np.clip(probs, 1e-12, None))).sum()
    return float(entropy / math.log(float(counts.size)))


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


def _sigmoid_np(values: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-np.clip(values, -60.0, 60.0)))).astype(np.float32)


def _raw_agreement_scores(raw_distance: np.ndarray,
                          selected_pos: np.ndarray,
                          topk: int,
                          margin: float) -> np.ndarray:
    """Soft score for whether selected targets are also raw-neighbour plausible."""
    if selected_pos.size == 0:
        return np.zeros(selected_pos.shape, dtype=np.float32)
    if int(topk) <= 0:
        return np.ones(selected_pos.shape, dtype=np.float32)

    k = min(max(1, int(topk)), raw_distance.shape[1])
    kth_distance = np.partition(raw_distance, kth=k - 1, axis=1)[:, k - 1]
    selected_raw_distance = np.take_along_axis(raw_distance, selected_pos, axis=1)
    scale = max(float(margin), 1e-6)
    return _sigmoid_np((kth_distance[:, None] - selected_raw_distance) / scale)


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
                     source_raw: torch.Tensor,
                     target_raw: torch.Tensor,
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
                     stability_rounds: int,
                     min_stability: int,
                     raw_blend: float,
                     raw_agreement_topk: int,
                     raw_agreement_margin: float,
                     raw_agreement_min: float,
                     raw_agreement_rank_weight: float,
                     seed: int) -> Tuple[GPRDRelation, Dict[str, float]]:
    """Build one stable reciprocal source -> target ranking teacher."""
    if source_fused.dim() != 2 or target_fused.dim() != 2:
        raise ValueError('GPRD expects [num_samples, feature_dim] tensors.')
    if source_fused.size(1) != target_fused.size(1):
        raise ValueError('source and target feature dims differ.')
    if source_raw.dim() != 2 or target_raw.dim() != 2:
        raise ValueError('GPRD raw features must be [num_samples, feature_dim] tensors.')
    if source_raw.size(0) != source_fused.size(0) or target_raw.size(0) != target_fused.size(0):
        raise ValueError('raw and fused feature counts differ.')
    if source_raw.size(1) != target_raw.size(1):
        raise ValueError('source and target raw feature dims differ.')

    device = source_fused.device
    dtype = source_fused.dtype
    raw_blend = min(1.0, max(0.0, float(raw_blend)))
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
    raw_agreement = torch.zeros((source_count, candidate_count),
                                dtype=dtype, device=device)
    confidence = torch.zeros(source_count, dtype=dtype, device=device)
    valid = torch.zeros(source_count, dtype=torch.bool, device=device)

    query_ids = _camera_balanced_indices(
        source_cameras, source_count, query_per_camera, seed)
    if query_ids.size == 0:
        relation = GPRDRelation(
            candidate_ids=candidate_ids.detach(),
            teacher_probs=teacher_probs.detach(),
            positive_mask=positive_mask.detach(),
            negative_mask=negative_mask.detach(),
            raw_agreement=raw_agreement.detach(),
            confidence=confidence.detach(),
            valid=valid.detach(),
            positive_count=positive_count,
            hard_negative_count=hard_negative_count,
        )
        return relation, {
            'coverage': 0.0,
            'query_count': float(query_ids.size),
            'gallery_count': 0.0,
            'teacher_entropy': 0.0,
            'active_rows': 0.0,
            'mean_confidence': 0.0,
            'mean_best_distance': 0.0,
            'mean_pos_neg_margin': 0.0,
            'mutual_positive_rate': 0.0,
            'reciprocal_rate': 0.0,
            'stable_edges': 0.0,
            'camera_balance_entropy': 0.0,
            'exact_nodes': float(query_ids.size),
            'candidate_count': float(candidate_count),
            'raw_blend': raw_blend,
            'mean_raw_agreement': 0.0,
            'raw_agreement_topk': float(raw_agreement_topk),
            'raw_agreement_min': float(raw_agreement_min),
        }

    source_np = _normalize_np(_to_numpy(source_fused)[query_ids])
    source_raw_np = _normalize_np(_to_numpy(source_raw)[query_ids])
    query_distance = _cosine_distance(source_np, source_np)
    stability_rounds = max(1, int(stability_rounds))
    min_stability = min(stability_rounds, max(1, int(min_stability)))

    stable_counts = [dict() for _ in range(query_ids.size)]
    stable_distance_sums = [dict() for _ in range(query_ids.size)]
    stable_agreement_sums = [dict() for _ in range(query_ids.size)]
    base_target_ids = None
    base_target_dist = None
    base_target_agree = None
    gallery_count = 0
    reciprocal_hits = 0.0
    reciprocal_total = 0.0
    camera_entropies = []

    for round_id in range(stability_rounds):
        round_seed = int(seed + 7919 + round_id * 104729)
        gallery_ids = _camera_balanced_indices(
            target_cameras, target_count, gallery_per_camera, round_seed)
        if gallery_ids.size == 0:
            continue
        gallery_count = max(gallery_count, int(gallery_ids.size))
        camera_entropies.append(_camera_balance_entropy(target_cameras, gallery_ids))

        target_np = _normalize_np(_to_numpy(target_fused)[gallery_ids])
        target_raw_np = _normalize_np(_to_numpy(target_raw)[gallery_ids])
        base_distance = _cosine_distance(source_np, target_np)
        raw_distance = _cosine_distance(source_raw_np, target_raw_np)
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
        if raw_blend > 0.0:
            final_distance = (
                (1.0 - raw_blend) * final_distance
                + raw_blend * _row_minmax(raw_distance)
            ).astype(np.float32, copy=False)

        selected_dist, selected_pos = _topk_smallest(final_distance, candidate_count)
        selected_agree = _raw_agreement_scores(
            raw_distance, selected_pos, raw_agreement_topk, raw_agreement_margin)
        selected_target_ids = gallery_ids[selected_pos]
        mutual_mask = _mutual_positive_mask(
            final_distance, selected_pos[:, :positive_count],
            positive_count, mutual_topk)
        if base_target_ids is None:
            base_target_ids = selected_target_ids
            base_target_dist = selected_dist
            base_target_agree = selected_agree

        reciprocal_hits += float(mutual_mask.sum())
        reciprocal_total += float(mutual_mask.size)
        for row in range(query_ids.size):
            cols = np.flatnonzero(mutual_mask[row])
            for col in cols.tolist():
                target_id = int(selected_target_ids[row, col])
                stable_counts[row][target_id] = stable_counts[row].get(target_id, 0) + 1
                stable_distance_sums[row][target_id] = (
                    stable_distance_sums[row].get(target_id, 0.0)
                    + float(selected_dist[row, col])
                )
                stable_agreement_sums[row][target_id] = (
                    stable_agreement_sums[row].get(target_id, 0.0)
                    + float(selected_agree[row, col])
                )

    if base_target_ids is None:
        relation = GPRDRelation(
            candidate_ids=candidate_ids.detach(),
            teacher_probs=teacher_probs.detach(),
            positive_mask=positive_mask.detach(),
            negative_mask=negative_mask.detach(),
            raw_agreement=raw_agreement.detach(),
            confidence=confidence.detach(),
            valid=valid.detach(),
            positive_count=positive_count,
            hard_negative_count=hard_negative_count,
        )
        return relation, {
            'coverage': 0.0,
            'query_count': float(query_ids.size),
            'gallery_count': 0.0,
            'teacher_entropy': 0.0,
            'active_rows': 0.0,
            'mean_confidence': 0.0,
            'mean_best_distance': 0.0,
            'mean_pos_neg_margin': 0.0,
            'mutual_positive_rate': 0.0,
            'reciprocal_rate': 0.0,
            'stable_edges': 0.0,
            'camera_balance_entropy': 0.0,
            'exact_nodes': float(query_ids.size),
            'candidate_count': float(candidate_count),
            'raw_blend': raw_blend,
            'mean_raw_agreement': 0.0,
            'raw_agreement_topk': float(raw_agreement_topk),
            'raw_agreement_min': float(raw_agreement_min),
        }

    selected_target_ids = np.full((query_ids.size, candidate_count), -1, dtype=np.int64)
    selected_dist = np.ones((query_ids.size, candidate_count), dtype=np.float32)
    selected_agreement = np.zeros((query_ids.size, candidate_count), dtype=np.float32)
    row_teacher = np.zeros((query_ids.size, candidate_count), dtype=np.float32)
    pos_mask_np = np.zeros((query_ids.size, candidate_count), dtype=bool)
    hard_negative_mask = np.zeros((query_ids.size, candidate_count), dtype=bool)
    norm_entropy = np.ones((query_ids.size,), dtype=np.float32)
    margin = np.zeros((query_ids.size,), dtype=np.float32)
    positive_agreement = np.zeros((query_ids.size,), dtype=np.float32)
    row_confidence = np.zeros((query_ids.size,), dtype=np.float32)
    row_valid = np.zeros((query_ids.size,), dtype=bool)
    stable_edges = 0

    for row in range(query_ids.size):
        stable = []
        for target_id, support in stable_counts[row].items():
            if int(support) >= min_stability:
                stable.append((
                    target_id,
                    stable_distance_sums[row][target_id] / float(support),
                    int(support),
                    stable_agreement_sums[row][target_id] / float(support),
                ))
        rank_weight = max(0.0, float(raw_agreement_rank_weight))
        stable.sort(key=lambda item: (
            item[1] + rank_weight * (1.0 - item[3]), -item[3]))
        positives = stable[:positive_count]
        if not positives:
            continue

        positive_ids = set()
        positive_dist = []
        support_scores = []
        agreement_scores = []
        for col, (target_id, distance, support, agreement) in enumerate(positives):
            selected_target_ids[row, col] = int(target_id)
            selected_dist[row, col] = float(distance)
            selected_agreement[row, col] = float(agreement)
            pos_mask_np[row, col] = True
            positive_ids.add(int(target_id))
            positive_dist.append(float(distance))
            support_scores.append(float(support) / float(stability_rounds))
            agreement_scores.append(float(agreement))
        stable_edges += len(positives)
        positive_agreement[row] = (
            float(np.mean(agreement_scores)) if agreement_scores else 0.0)

        neg_col = positive_count
        for base_col in range(positive_count, base_target_ids.shape[1]):
            if neg_col >= candidate_count:
                break
            target_id = int(base_target_ids[row, base_col])
            if target_id < 0 or target_id in positive_ids:
                continue
            selected_target_ids[row, neg_col] = target_id
            selected_dist[row, neg_col] = float(base_target_dist[row, base_col])
            selected_agreement[row, neg_col] = float(base_target_agree[row, base_col])
            hard_negative_mask[row, neg_col] = True
            neg_col += 1
        if neg_col <= positive_count:
            continue

        candidate_dist = selected_dist[row, :neg_col].reshape(1, -1)
        candidate_probs = _softmax_negative(candidate_dist, teacher_temperature).reshape(-1)
        row_teacher[row, :neg_col] = candidate_probs
        entropy = -(
            candidate_probs * np.log(np.clip(candidate_probs, 1e-12, None))
        ).sum()
        norm_entropy[row] = float(entropy / math.log(float(max(2, neg_col))))
        margin[row] = float(selected_dist[row, positive_count] - selected_dist[row, 0])
        best_score = 1.0 - np.clip(selected_dist[row, 0], 0.0, 1.0)
        support_score = float(np.mean(support_scores)) if support_scores else 0.0
        row_confidence[row] = (
            0.20 * best_score
            + 0.30 * (1.0 - norm_entropy[row])
            + 0.15 * np.clip(margin[row] / 0.15, 0.0, 1.0)
            + 0.15 * support_score
            + 0.20 * positive_agreement[row]
        )
        row_valid[row] = (
            (row_confidence[row] >= float(confidence_floor))
            and (norm_entropy[row] <= float(entropy_ceiling))
            and (positive_agreement[row] >= float(raw_agreement_min))
        )

    q_tensor = torch.from_numpy(query_ids).to(device=device, dtype=torch.long)
    candidate_ids[q_tensor] = torch.from_numpy(selected_target_ids).to(
        device=device, dtype=torch.long)
    teacher_probs[q_tensor] = torch.from_numpy(row_teacher).to(device=device, dtype=dtype)

    positive_mask[q_tensor] = torch.from_numpy(pos_mask_np).to(device=device)
    negative_mask[q_tensor] = torch.from_numpy(hard_negative_mask).to(device=device)
    raw_agreement[q_tensor] = torch.from_numpy(selected_agreement).to(
        device=device, dtype=dtype)
    confidence[q_tensor] = torch.from_numpy(row_confidence).to(device=device, dtype=dtype)
    valid[q_tensor] = torch.from_numpy(row_valid).to(device=device, dtype=torch.bool)

    valid_count = int(row_valid.sum())
    relation = GPRDRelation(
        candidate_ids=candidate_ids.detach(),
        teacher_probs=teacher_probs.detach(),
        positive_mask=positive_mask.detach(),
        negative_mask=negative_mask.detach(),
        raw_agreement=raw_agreement.detach(),
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
        'mean_best_distance': float(selected_dist[row_valid, 0].mean())
        if valid_count > 0 else 0.0,
        'mean_pos_neg_margin': float(margin[row_valid].mean())
        if valid_count > 0 else 0.0,
        'mean_raw_agreement': float(positive_agreement[row_valid].mean())
        if valid_count > 0 else 0.0,
        'mutual_positive_rate': (float(reciprocal_hits) /
                                 float(max(1.0, reciprocal_total))),
        'reciprocal_rate': (float(reciprocal_hits) /
                            float(max(1.0, reciprocal_total))),
        'stable_edges': float(stable_edges),
        'camera_balance_entropy': float(np.mean(camera_entropies))
        if camera_entropies else 0.0,
        'exact_nodes': float(query_ids.size + gallery_count),
        'candidate_count': float(candidate_count),
        'raw_blend': raw_blend,
        'raw_agreement_topk': float(raw_agreement_topk),
        'raw_agreement_min': float(raw_agreement_min),
    }
    return relation, metrics


@torch.no_grad()
def build_gprd_teacher(rgb_features: torch.Tensor,
                       ir_features: torch.Tensor,
                       rgb_view2: Optional[torch.Tensor] = None,
                       ir_view2: Optional[torch.Tensor] = None,
                       rgb_cameras: Optional[torch.Tensor] = None,
                       ir_cameras: Optional[torch.Tensor] = None,
                       global_weight: float = 0.25,
                       query_per_camera: int = 1536,
                       gallery_per_camera: int = 3072,
                       positive_count: int = 2,
                       hard_negative_count: int = 14,
                       mutual_topk: int = 20,
                       rerank_k1: int = 30,
                       rerank_k2: int = 3,
                       rerank_lambda: float = 0.10,
                       csls_neighbors: int = 5,
                       csls_blend: float = 0.75,
                       teacher_temperature: float = 0.08,
                       confidence_floor: float = 0.25,
                       entropy_ceiling: float = 0.92,
                       stability_rounds: int = 2,
                       min_stability: int = 1,
                       raw_blend: float = 0.20,
                       raw_agreement_topk: int = 256,
                       raw_agreement_margin: float = 0.05,
                       raw_agreement_min: float = 0.20,
                       raw_agreement_rank_weight: float = 0.04,
                       seed: int = 1) -> Tuple[GPRDRelation, GPRDRelation, Dict[str, float]]:
    """Build RGB->IR and IR->RGB stable reciprocal ranking teachers."""
    if rgb_view2 is None:
        rgb_view2 = rgb_features
    if ir_view2 is None:
        ir_view2 = ir_features

    rgb_fused = fuse_metric_features(rgb_features, rgb_view2, global_weight)
    ir_fused = fuse_metric_features(ir_features, ir_view2, global_weight)

    rgb_to_ir, rgb_stats = _build_direction(
        rgb_fused, ir_fused, rgb_features, ir_features, rgb_cameras, ir_cameras,
        query_per_camera, gallery_per_camera, positive_count,
        hard_negative_count, mutual_topk, rerank_k1, rerank_k2,
        rerank_lambda, csls_neighbors, csls_blend, teacher_temperature,
        confidence_floor, entropy_ceiling, stability_rounds, min_stability,
        raw_blend, raw_agreement_topk, raw_agreement_margin,
        raw_agreement_min, raw_agreement_rank_weight, seed)
    ir_to_rgb, ir_stats = _build_direction(
        ir_fused, rgb_fused, ir_features, rgb_features, ir_cameras, rgb_cameras,
        query_per_camera, gallery_per_camera, positive_count,
        hard_negative_count, mutual_topk, rerank_k1, rerank_k2,
        rerank_lambda, csls_neighbors, csls_blend, teacher_temperature,
        confidence_floor, entropy_ceiling, stability_rounds, min_stability,
        raw_blend, raw_agreement_topk, raw_agreement_margin,
        raw_agreement_min, raw_agreement_rank_weight, seed + 104729)

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
        'mean_raw_agreement': 0.5 * (
            rgb_stats['mean_raw_agreement'] + ir_stats['mean_raw_agreement']),
        'mutual_positive_rate': 0.5 * (
            rgb_stats['mutual_positive_rate'] + ir_stats['mutual_positive_rate']),
        'reciprocal_rate': 0.5 * (
            rgb_stats['reciprocal_rate'] + ir_stats['reciprocal_rate']),
        'stable_edges': rgb_stats['stable_edges'] + ir_stats['stable_edges'],
        'stable_edges_rgb': rgb_stats['stable_edges'],
        'stable_edges_ir': ir_stats['stable_edges'],
        'camera_balance_entropy': 0.5 * (
            rgb_stats['camera_balance_entropy'] +
            ir_stats['camera_balance_entropy']),
        'stability_rounds': float(max(1, int(stability_rounds))),
        'min_stability': float(max(1, int(min_stability))),
        'raw_blend': min(1.0, max(0.0, float(raw_blend))),
        'raw_agreement_topk': float(raw_agreement_topk),
        'raw_agreement_min': float(raw_agreement_min),
        'raw_agreement_rank_weight': float(max(0.0, raw_agreement_rank_weight)),
    }
    return rgb_to_ir, ir_to_rgb, metrics


def gprd_ranking_loss(student_features: torch.Tensor,
                      source_indices: torch.Tensor,
                      target_memory: torch.Tensor,
                      relation: Optional[GPRDRelation],
                      temperature: float = 0.08,
                      margin: float = 0.04,
                      pairwise_weight: float = 0.50,
                      agreement_floor: float = 0.35) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Distill a soft reciprocal neighbourhood into the current raw embedding."""
    zero = student_features.sum() * 0.0
    if relation is None or source_indices.numel() == 0:
        return zero, {
            'active_rows': 0.0,
            'mean_confidence': 0.0,
            'pull': 0.0,
            'margin': 0.0,
            'listwise': 0.0,
            'pairwise': 0.0,
            'pos_sim': 0.0,
            'neg_sim': 0.0,
            'raw_support': 0.0,
            'raw_agreement': 0.0,
            'gap': 0.0,
        }

    source_indices = source_indices.long().view(-1)
    if source_indices.device != relation.candidate_ids.device:
        source_indices = source_indices.to(relation.candidate_ids.device)
    candidate_ids = relation.candidate_ids.index_select(0, source_indices)
    teacher_probs = relation.teacher_probs.index_select(0, source_indices)
    positive_mask = relation.positive_mask.index_select(0, source_indices)
    negative_mask = relation.negative_mask.index_select(0, source_indices)
    raw_agreement = relation.raw_agreement.index_select(0, source_indices)
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
            'pull': 0.0,
            'margin': 0.0,
            'listwise': 0.0,
            'pairwise': 0.0,
            'pos_sim': 0.0,
            'neg_sim': 0.0,
            'raw_support': 0.0,
            'raw_agreement': 0.0,
            'gap': 0.0,
        }

    target_memory = F.normalize(target_memory.detach(), dim=1)
    safe_ids = candidate_ids.clamp_min(0)
    candidate_features = target_memory[safe_ids]
    student = F.normalize(student_features, dim=1)
    sim = torch.bmm(candidate_features, student.unsqueeze(2)).squeeze(2)
    sim = sim.masked_fill(~candidate_valid, -10000.0)
    tau = max(float(temperature), 1e-6)

    target = teacher_probs * candidate_valid.to(teacher_probs.dtype)
    target = target / target.sum(dim=1, keepdim=True).clamp_min(1e-12)
    logits = sim / tau
    log_probs = F.log_softmax(logits, dim=1)
    listwise = -(target * log_probs).sum(dim=1)

    positive_target = target * positive_mask.to(target.dtype)
    positive_target = positive_target / positive_target.sum(dim=1, keepdim=True).clamp_min(1e-12)
    positive_agreement = (
        raw_agreement * positive_mask.to(raw_agreement.dtype)
    ).sum(dim=1) / positive_mask.sum(dim=1).clamp_min(1).to(raw_agreement.dtype)
    pos_sim_each = sim.masked_fill(~positive_mask, 0.0)
    pos_anchor = (pos_sim_each * positive_target).sum(dim=1)
    neg_sim = sim.masked_fill(~negative_mask, -10000.0)
    hard_neg = tau * torch.logsumexp(neg_sim / tau, dim=1)
    gap = pos_anchor - hard_neg
    raw_support = torch.sigmoid((gap + 0.08) / 0.05).detach()
    soft_guard = (0.40 + 0.60 * raw_support).detach()
    agree_floor = min(1.0, max(0.0, float(agreement_floor)))
    agreement_guard = (
        agree_floor + (1.0 - agree_floor) * positive_agreement.detach()
    )
    pull = F.softplus((0.55 - pos_anchor) / tau) * tau
    margin_loss = F.softplus((hard_neg - pos_anchor + float(margin)) / tau) * tau
    pos_spread = (
        ((sim - pos_anchor.unsqueeze(1)).pow(2) * positive_target).sum(dim=1)
    )

    pairwise_scale = (0.50 + 0.50 * positive_agreement.detach())
    weights = confidence * soft_guard * agreement_guard * row_valid.to(confidence.dtype)
    per_row = (
        listwise
        + float(pairwise_weight) * pairwise_scale * margin_loss
        + 0.10 * pull
        + 0.02 * pos_spread
    )
    loss = (per_row * weights).sum() / weights.sum().clamp_min(1e-12)
    active = row_valid.sum().item()
    return loss, {
        'active_rows': float(active),
        'mean_confidence': float(weights[row_valid].mean().item()),
        'pull': float(pull[row_valid].mean().item()),
        'margin': float(margin_loss[row_valid].mean().item()),
        'listwise': float(listwise[row_valid].mean().item()),
        'pairwise': float(margin_loss[row_valid].mean().item()),
        'pos_sim': float(pos_anchor[row_valid].mean().item()),
        'neg_sim': float(hard_neg[row_valid].mean().item()),
        'raw_support': float(raw_support[row_valid].mean().item()),
        'raw_agreement': float(positive_agreement[row_valid].mean().item()),
        'gap': float(gap[row_valid].mean().item()),
    }
