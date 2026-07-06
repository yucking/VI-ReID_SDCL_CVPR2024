"""Metric-Stack Retrieval Distillation (MSRD).

This module turns the validated SYSU Metric Stack signal into a training-only
teacher.  It uses only unlabeled training features:

* global/local feature fusion with the fixed Metric Stack weight;
* CSLS-style cross-domain local scaling to suppress hubs;
* a k-reciprocal/Jaccard neighbourhood proxy over source and target memories;
* a listwise distribution over many opposite-modality candidates.

The student still sees the original raw/global feature memory in the loss, so
the objective is to distill the stronger neighbourhood relation into the raw
embedding space instead of changing test-time evaluation.
"""

from __future__ import absolute_import

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class MSRDRelation(object):
    """Detached metric-stack candidate ranking targets."""

    candidate_ids: torch.Tensor
    teacher_probs: torch.Tensor
    confidence: torch.Tensor
    valid: torch.Tensor
    stable_count: torch.Tensor
    positive_count: int
    hard_negative_count: int


def _normalize(features: torch.Tensor) -> torch.Tensor:
    return F.normalize(features, dim=1)


def fuse_metric_features(global_features: torch.Tensor,
                         local_features: torch.Tensor,
                         global_weight: float) -> torch.Tensor:
    """Match the fixed Metric Stack global/local fusion."""
    if not 0.0 < float(global_weight) < 1.0:
        raise ValueError('global_weight must be in (0, 1).')
    local_weight = 1.0 - float(global_weight)
    return torch.cat((
        math.sqrt(float(global_weight)) * _normalize(global_features),
        math.sqrt(local_weight) * _normalize(local_features),
    ), dim=1)


@torch.no_grad()
def _chunked_topk(query_features: torch.Tensor,
                  gallery_features: torch.Tensor,
                  topk: int,
                  chunk_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return cosine top-k values and indices without materialising all rows."""
    if query_features.numel() == 0 or gallery_features.numel() == 0:
        values = query_features.new_empty((query_features.size(0), 0))
        ids = torch.empty_like(values, dtype=torch.long)
        return values, ids
    k = min(int(topk), int(gallery_features.size(0)))
    query = _normalize(query_features)
    gallery = _normalize(gallery_features)
    all_values = []
    all_indices = []
    step = max(1, int(chunk_size))
    for start in range(0, query.size(0), step):
        scores = query[start:start + step].mm(gallery.t())
        values, indices = torch.topk(scores, k=k, dim=1, largest=True,
                                     sorted=True)
        all_values.append(values)
        all_indices.append(indices)
    return torch.cat(all_values, dim=0), torch.cat(all_indices, dim=0)


@torch.no_grad()
def _self_topk_ids(features: torch.Tensor, topk: int,
                  chunk_size: int) -> torch.Tensor:
    """Top-k same-domain neighbours excluding self."""
    k = min(int(topk) + 1, int(features.size(0)))
    _, ids = _chunked_topk(features, features, k, chunk_size)
    rows = torch.arange(features.size(0), device=ids.device).view(-1, 1)
    cleaned = []
    for row in range(ids.size(0)):
        keep = ids[row][ids[row] != rows[row, 0]][:int(topk)]
        if keep.numel() < int(topk):
            pad = keep.new_full((int(topk) - keep.numel(),), -1)
            keep = torch.cat((keep, pad), dim=0)
        cleaned.append(keep.view(1, -1))
    return torch.cat(cleaned, dim=0)


def _row_minmax(values: torch.Tensor) -> torch.Tensor:
    minimum = values.min(dim=1, keepdim=True)[0]
    maximum = values.max(dim=1, keepdim=True)[0]
    return (values - minimum) / (maximum - minimum).clamp_min(1e-12)


def _jaccard_affinity(left, right) -> float:
    if not isinstance(left, set):
        left = set(int(x) for x in left if int(x) >= 0)
    if not isinstance(right, set):
        right = set(int(x) for x in right if int(x) >= 0)
    if not left or not right:
        return 0.0
    union = len(left | right)
    if union <= 0:
        return 0.0
    return float(len(left & right)) / float(union)


@torch.no_grad()
def _build_direction(source_global: torch.Tensor,
                     source_local: torch.Tensor,
                     target_global: torch.Tensor,
                     target_local: torch.Tensor,
                     global_weight: float,
                     candidate_pool: int,
                     candidate_count: int,
                     rerank_k1: int,
                     rerank_k2: int,
                     rerank_lambda: float,
                     csls_neighbors: int,
                     csls_blend: float,
                     teacher_temperature: float,
                     confidence_floor: float,
                     chunk_size: int) -> Tuple[MSRDRelation, Dict[str, float]]:
    """Build source -> target metric-stack teacher."""
    if source_global.dim() != 2 or target_global.dim() != 2:
        raise ValueError('MSRD expects [num_samples, feature_dim] tensors.')
    if source_global.size(1) != target_global.size(1):
        raise ValueError('MSRD source and target global feature dims differ.')
    if source_local.size(0) != source_global.size(0):
        raise ValueError('MSRD source global/local feature counts differ.')
    if target_local.size(0) != target_global.size(0):
        raise ValueError('MSRD target global/local feature counts differ.')

    device = source_global.device
    target_global = target_global.to(device)
    source_local = source_local.to(device)
    target_local = target_local.to(device)

    source_fused = fuse_metric_features(source_global, source_local,
                                        global_weight)
    target_fused = fuse_metric_features(target_global, target_local,
                                        global_weight)

    pool = max(int(candidate_pool), int(candidate_count), int(rerank_k1))
    pool = min(pool, int(target_fused.size(0)))
    count = min(int(candidate_count), pool)
    k1 = max(1, min(int(rerank_k1), pool))
    k2 = max(1, int(rerank_k2))

    forward_scores, forward_ids = _chunked_topk(
        source_fused, target_fused, pool, chunk_size)
    reverse_scores, reverse_ids = _chunked_topk(
        target_fused, source_fused, k1, chunk_size)
    self_k = min(k1 + k2, max(1, int(max(source_fused.size(0), target_fused.size(0))) - 1))
    source_self_ids = _self_topk_ids(source_fused, self_k, chunk_size)
    target_self_ids = _self_topk_ids(target_fused, self_k, chunk_size)

    csls_k = max(1, int(csls_neighbors))
    source_scale = forward_scores[:, :min(csls_k, forward_scores.size(1))].mean(dim=1)
    target_scale = reverse_scores[:, :min(csls_k, reverse_scores.size(1))].mean(dim=1)

    forward_ids_cpu = forward_ids.detach().cpu()
    forward_scores_cpu = forward_scores.detach().cpu()
    reverse_ids_cpu = reverse_ids.detach().cpu()
    source_self_cpu = source_self_ids.detach().cpu()
    target_self_cpu = target_self_ids.detach().cpu()
    source_cross_sets = [
        set(int(x) for x in forward_ids_cpu[row, :k1].tolist() if int(x) >= 0)
        for row in range(forward_ids_cpu.size(0))
    ]
    source_self_sets = [
        set(int(x) for x in source_self_cpu[row, :self_k].tolist() if int(x) >= 0)
        for row in range(source_self_cpu.size(0))
    ]
    target_cross_sets = [
        set(int(x) for x in reverse_ids_cpu[row, :k1].tolist() if int(x) >= 0)
        for row in range(reverse_ids_cpu.size(0))
    ]
    target_self_sets = [
        set(int(x) for x in target_self_cpu[row, :self_k].tolist() if int(x) >= 0)
        for row in range(target_self_cpu.size(0))
    ]

    candidate_ids = torch.full((source_global.size(0), count), -1,
                               dtype=torch.long, device=device)
    teacher_probs = torch.zeros((source_global.size(0), count),
                                dtype=source_global.dtype, device=device)
    confidence = torch.zeros(source_global.size(0), dtype=source_global.dtype,
                             device=device)
    valid = torch.zeros(source_global.size(0), dtype=torch.bool, device=device)
    stable_count = torch.zeros(source_global.size(0), dtype=torch.long,
                               device=device)

    rerank_dist_cpu = torch.empty((source_global.size(0), pool),
                                  dtype=torch.float32)
    reciprocal_edges = 0
    neighbour_edges = 0

    for source_id in range(source_global.size(0)):
        source_cross = source_cross_sets[source_id]
        source_self = source_self_sets[source_id]
        for rank in range(pool):
            target_id = int(forward_ids_cpu[source_id, rank])
            target_cross = target_cross_sets[target_id]
            target_self = target_self_sets[target_id]
            reciprocal = 1.0 if source_id in target_cross else 0.0
            if reciprocal > 0.0:
                reciprocal_edges += 1
            # Two same-domain neighbourhood overlaps: target-candidate
            # agreement and source-candidate agreement.  k2 controls how much
            # expanded neighbourhood is considered, matching rerank's second
            # smoothing knob without materialising a full train-size matrix.
            j_target = _jaccard_affinity(source_cross, target_self)
            j_source = _jaccard_affinity(source_self, target_cross)
            neighbour = 0.5 * (j_target + j_source)
            if neighbour > 0.0:
                neighbour_edges += 1
            affinity = min(1.0, neighbour + 0.25 * reciprocal)
            jaccard_dist = 1.0 - affinity
            base_dist = 1.0 - float(forward_scores_cpu[source_id, rank])
            rerank_dist_cpu[source_id, rank] = (
                float(rerank_lambda) * base_dist
                + (1.0 - float(rerank_lambda)) * jaccard_dist)

    rerank_dist = rerank_dist_cpu.to(device=device, dtype=source_global.dtype)
    candidate_csls = -(2.0 * forward_scores
                       - source_scale[:, None]
                       - target_scale[forward_ids])
    final_dist = (float(csls_blend) * _row_minmax(rerank_dist)
                  + (1.0 - float(csls_blend)) * _row_minmax(candidate_csls))
    selected_dist, selected_pos = torch.topk(final_dist, k=count, dim=1,
                                             largest=False, sorted=True)
    selected_ids = torch.gather(forward_ids, 1, selected_pos)
    candidate_ids.copy_(selected_ids)
    probabilities = F.softmax(
        -selected_dist / max(float(teacher_temperature), 1e-6), dim=1)
    teacher_probs.copy_(probabilities.detach())

    best_score = 1.0 - selected_dist[:, 0].clamp(0.0, 1.0)
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=1)
    norm_entropy = entropy / math.log(float(max(2, count)))
    row_confidence = (0.5 * best_score + 0.5 * (1.0 - norm_entropy)).clamp(0.0, 1.0)
    confidence.copy_(row_confidence)
    valid.copy_(row_confidence >= float(confidence_floor))
    stable_count.copy_((selected_dist < selected_dist[:, :1] + 0.15).sum(dim=1))

    valid_count = int(valid.sum().item())
    relation = MSRDRelation(
        candidate_ids=candidate_ids.detach(),
        teacher_probs=teacher_probs.detach(),
        confidence=confidence.detach(),
        valid=valid.detach(),
        stable_count=stable_count.detach(),
        positive_count=int(candidate_count),
        hard_negative_count=0,
    )
    metrics = {
        'coverage': float(valid_count) / float(max(1, source_global.size(0))),
        'reciprocal_rate': float(reciprocal_edges) / float(
            max(1, source_global.size(0) * pool)),
        'neighbour_rate': float(neighbour_edges) / float(
            max(1, source_global.size(0) * pool)),
        'teacher_entropy': float(norm_entropy[valid].mean().item())
        if valid_count > 0 else 0.0,
        'active_rows': float(valid_count),
        'mean_confidence': float(confidence[valid].mean().item())
        if valid_count > 0 else 0.0,
        'mean_best_distance': float(selected_dist[:, 0].mean().item()),
    }
    return relation, metrics


@torch.no_grad()
def build_msrd_teacher(rgb_features: torch.Tensor,
                       ir_features: torch.Tensor,
                       rgb_view2: Optional[torch.Tensor] = None,
                       ir_view2: Optional[torch.Tensor] = None,
                       global_weight: float = 0.25,
                       candidate_pool: int = 96,
                       candidate_count: int = 64,
                       rerank_k1: int = 30,
                       rerank_k2: int = 3,
                       rerank_lambda: float = 0.10,
                       csls_neighbors: int = 5,
                       csls_blend: float = 0.75,
                       teacher_temperature: float = 0.07,
                       confidence_floor: float = 0.0,
                       chunk_size: int = 1024) -> Tuple[MSRDRelation, MSRDRelation, Dict[str, float]]:
    """Build RGB->IR and IR->RGB Metric-Stack teachers."""
    if rgb_view2 is None:
        rgb_view2 = rgb_features
    if ir_view2 is None:
        ir_view2 = ir_features
    rgb_to_ir, rgb_stats = _build_direction(
        rgb_features, rgb_view2, ir_features, ir_view2, global_weight,
        candidate_pool, candidate_count, rerank_k1, rerank_k2,
        rerank_lambda, csls_neighbors, csls_blend, teacher_temperature,
        confidence_floor, chunk_size)
    ir_to_rgb, ir_stats = _build_direction(
        ir_features, ir_view2, rgb_features, rgb_view2, global_weight,
        candidate_pool, candidate_count, rerank_k1, rerank_k2,
        rerank_lambda, csls_neighbors, csls_blend, teacher_temperature,
        confidence_floor, chunk_size)
    metrics = {
        'coverage_rgb': rgb_stats['coverage'],
        'coverage_ir': ir_stats['coverage'],
        'reciprocal_rate': 0.5 * (
            rgb_stats['reciprocal_rate'] + ir_stats['reciprocal_rate']),
        'neighbour_rate': 0.5 * (
            rgb_stats['neighbour_rate'] + ir_stats['neighbour_rate']),
        'teacher_entropy': 0.5 * (
            rgb_stats['teacher_entropy'] + ir_stats['teacher_entropy']),
        'active_rows_rgb': rgb_stats['active_rows'],
        'active_rows_ir': ir_stats['active_rows'],
        'mean_confidence': 0.5 * (
            rgb_stats['mean_confidence'] + ir_stats['mean_confidence']),
        'mean_best_distance': 0.5 * (
            rgb_stats['mean_best_distance'] + ir_stats['mean_best_distance']),
    }
    return rgb_to_ir, ir_to_rgb, metrics


def msrd_listwise_loss(student_features: torch.Tensor,
                       source_indices: torch.Tensor,
                       target_memory: torch.Tensor,
                       relation: Optional[MSRDRelation],
                       temperature: float = 0.05) -> Tuple[torch.Tensor, Dict[str, float]]:
    """KL/ListNet loss from Metric-Stack teacher to raw student features."""
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
    logits = torch.bmm(candidate_features, student.unsqueeze(2)).squeeze(2)
    logits = logits / max(float(temperature), 1e-6)
    logits = logits.masked_fill(~candidate_valid, -1e4)
    log_probs = F.log_softmax(logits, dim=1)
    target = teacher_probs * candidate_valid.to(teacher_probs.dtype)
    target = target / target.sum(dim=1, keepdim=True).clamp_min(1e-12)
    kl = (target * (target.clamp_min(1e-12).log() - log_probs)).sum(dim=1)
    weights = confidence * row_valid.to(confidence.dtype)
    denominator = weights.sum().clamp_min(1e-12)
    loss = (kl * weights).sum() / denominator
    return loss, {
        'active_rows': float(row_valid.sum().item()),
        'mean_confidence': float(weights[row_valid].mean().item()),
    }
