"""Metric-Stack Pseudo-Label Fusion for SYSU Stage-2 training.

This module uses the strong test-time Metric Stack signal at the supervision
point where it matters most: pseudo labels.  It builds a full RGB+IR graph,
computes exact k-reciprocal distances, converts sample neighbours into
cluster-level cross-modal matches, and returns one unified pseudo-label space.

It does not add an auxiliary loss and it never treats close reranked neighbours
as negatives.
"""

from __future__ import absolute_import

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class MSPLFMerge(object):
    rgb_cluster: int
    ir_cluster: int
    distance: float
    margin: float
    rgb_support: int
    ir_support: int


def _normalize_torch(features: torch.Tensor) -> torch.Tensor:
    return F.normalize(features, dim=1)


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


def _metric_stack_full_cross(rgb_features: np.ndarray,
                             ir_features: np.ndarray,
                             rerank_k1: int,
                             rerank_k2: int,
                             rerank_lambda: float) -> Tuple[np.ndarray, np.ndarray]:
    """Run exact k-reciprocal reranking once on the full RGB+IR graph."""
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


def _valid_labels(labels: np.ndarray) -> np.ndarray:
    values = np.array(sorted(int(v) for v in set(labels.tolist()) if int(v) != -1),
                      dtype=np.int64)
    return values


def _label_to_local(labels: np.ndarray, keys: np.ndarray) -> np.ndarray:
    mapper = {int(label): idx for idx, label in enumerate(keys.tolist())}
    local = np.full(labels.shape[0], -1, dtype=np.int64)
    for idx, label in enumerate(labels.tolist()):
        local[idx] = mapper.get(int(label), -1)
    return local


def _aggregate_cluster_distance(distance: np.ndarray,
                                source_labels: np.ndarray,
                                target_labels: np.ndarray,
                                source_keys: np.ndarray,
                                target_keys: np.ndarray,
                                sample_topk: int,
                                min_pairs: int) -> Tuple[np.ndarray, np.ndarray]:
    source_local = _label_to_local(source_labels, source_keys)
    target_local = _label_to_local(target_labels, target_keys)
    sample_topk = min(max(1, int(sample_topk)), distance.shape[1])
    if sample_topk == distance.shape[1]:
        candidate_ids = np.argsort(distance, axis=1)[:, :sample_topk]
    else:
        candidate_ids = np.argpartition(distance, kth=sample_topk - 1, axis=1)[:, :sample_topk]
    candidate_values = np.take_along_axis(distance, candidate_ids, axis=1)

    row_ids = np.repeat(np.arange(distance.shape[0], dtype=np.int64), sample_topk)
    source_ids = source_local[row_ids]
    target_ids = target_local[candidate_ids.reshape(-1)]
    values = candidate_values.reshape(-1).astype(np.float64, copy=False)
    mask = (source_ids >= 0) & (target_ids >= 0)

    sums = np.zeros((len(source_keys), len(target_keys)), dtype=np.float64)
    counts = np.zeros((len(source_keys), len(target_keys)), dtype=np.int32)
    np.add.at(sums, (source_ids[mask], target_ids[mask]), values[mask])
    np.add.at(counts, (source_ids[mask], target_ids[mask]), 1)

    cluster_distance = np.full(sums.shape, np.inf, dtype=np.float32)
    valid = counts >= int(min_pairs)
    cluster_distance[valid] = (sums[valid] / counts[valid]).astype(np.float32)
    return cluster_distance, counts


def _top_indices(row: np.ndarray, topk: int) -> np.ndarray:
    finite = np.where(np.isfinite(row))[0]
    if finite.size == 0:
        return finite
    order = finite[np.argsort(row[finite])]
    return order[:min(int(topk), order.size)]


def _margin_for_pair(row: np.ndarray, index: int) -> float:
    finite = np.where(np.isfinite(row))[0]
    if finite.size <= 1:
        return float('inf')
    ordered = finite[np.argsort(row[finite])]
    if int(ordered[0]) == int(index):
        return float(row[ordered[1]] - row[index])
    return float(row[ordered[0]] - row[index])


def _row_normalize(weights: np.ndarray) -> np.ndarray:
    normalized = np.zeros_like(weights, dtype=np.float32)
    row_sum = weights.sum(axis=1, keepdims=True)
    np.divide(weights, np.clip(row_sum, 1e-12, None),
              out=normalized, where=row_sum > 0.0)
    return normalized


def _cluster_affinity(distance: np.ndarray,
                      topk: int,
                      max_distance: float,
                      temperature: float) -> np.ndarray:
    topk = max(1, int(topk))
    temperature = max(float(temperature), 1e-6)
    affinity = np.zeros_like(distance, dtype=np.float32)
    for row_id in range(distance.shape[0]):
        row = distance[row_id]
        finite = np.where(np.isfinite(row))[0]
        if finite.size == 0:
            continue
        finite = finite[row[finite] <= float(max_distance)]
        if finite.size == 0:
            continue
        ordered = finite[np.argsort(row[finite])]
        selected = ordered[:min(topk, ordered.size)]
        values = row[selected].astype(np.float32, copy=False)
        logits = -(values - values.min()) / temperature
        logits = logits - logits.max()
        weights = np.exp(logits).astype(np.float32)
        affinity[row_id, selected] = weights / np.clip(weights.sum(), 1e-12, None)
    return affinity


def _top_score_indices(row: np.ndarray, topk: int) -> np.ndarray:
    positive = np.where(row > 0.0)[0]
    if positive.size == 0:
        return positive
    order = positive[np.argsort(-row[positive])]
    return order[:min(int(topk), order.size)]


def _diffuse_cluster_affinity(rgb_to_ir_cluster: np.ndarray,
                              ir_to_rgb_cluster: np.ndarray,
                              rgb_cluster_topk: int,
                              ir_cluster_topk: int,
                              max_distance: float,
                              neighborhood_blend: float,
                              neighborhood_temperature: float
                              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    direct_rgb_ir = _cluster_affinity(
        rgb_to_ir_cluster, rgb_cluster_topk, max_distance,
        neighborhood_temperature)
    reverse_ir_rgb = _cluster_affinity(
        ir_to_rgb_cluster, ir_cluster_topk, max_distance,
        neighborhood_temperature)
    propagated = np.matmul(np.matmul(direct_rgb_ir, reverse_ir_rgb),
                           direct_rgb_ir).astype(np.float32, copy=False)
    propagated = _row_normalize(propagated)
    blend = min(max(float(neighborhood_blend), 0.0), 1.0)
    fused = ((1.0 - blend) * direct_rgb_ir + blend * propagated).astype(
        np.float32, copy=False)
    return _row_normalize(fused), direct_rgb_ir, propagated


class _UnionFind(object):
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: int, right: int) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left != root_right:
            if root_left < root_right:
                self.parent[root_right] = root_left
            else:
                self.parent[root_left] = root_right


def _relabel_from_merges(rgb_labels: np.ndarray,
                         ir_labels: np.ndarray,
                         rgb_keys: np.ndarray,
                         ir_keys: np.ndarray,
                         merges: Sequence[MSPLFMerge]) -> Tuple[np.ndarray, np.ndarray, int]:
    rgb_local_by_label = {int(label): idx for idx, label in enumerate(rgb_keys.tolist())}
    ir_local_by_label = {int(label): idx for idx, label in enumerate(ir_keys.tolist())}
    offset = len(rgb_keys)
    uf = _UnionFind(len(rgb_keys) + len(ir_keys))
    for merge in merges:
        uf.union(rgb_local_by_label[int(merge.rgb_cluster)],
                 offset + ir_local_by_label[int(merge.ir_cluster)])

    component_to_new: Dict[int, int] = {}
    for node in range(len(rgb_keys) + len(ir_keys)):
        root = uf.find(node)
        if root not in component_to_new:
            component_to_new[root] = len(component_to_new)

    rgb_map = {
        int(label): component_to_new[uf.find(local)]
        for local, label in enumerate(rgb_keys.tolist())
    }
    ir_map = {
        int(label): component_to_new[uf.find(offset + local)]
        for local, label in enumerate(ir_keys.tolist())
    }

    new_rgb = np.array([rgb_map.get(int(label), -1) for label in rgb_labels.tolist()],
                       dtype=np.int64)
    new_ir = np.array([ir_map.get(int(label), -1) for label in ir_labels.tolist()],
                     dtype=np.int64)
    return new_rgb, new_ir, len(component_to_new)


@torch.no_grad()
def build_msplf_pseudo_labels(rgb_features: torch.Tensor,
                              ir_features: torch.Tensor,
                              rgb_labels: Iterable[int],
                              ir_labels: Iterable[int],
                              rgb_view2: torch.Tensor = None,
                              ir_view2: torch.Tensor = None,
                              global_weight: float = 0.25,
                              sample_topk: int = 96,
                              rgb_cluster_topk: int = 3,
                              ir_cluster_topk: int = 8,
                              min_pairs: int = 2,
                              max_distance: float = 0.22,
                              min_margin: float = -0.02,
                              rerank_k1: int = 30,
                              rerank_k2: int = 3,
                              rerank_lambda: float = 0.10,
                              csls_neighbors: int = 5,
                              csls_blend: float = 0.75,
                              neighborhood_blend: float = 0.40,
                              neighborhood_temperature: float = 0.07
                              ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Return IR-anchor pseudo labels built from the full Metric Stack graph.

    IR labels are kept as the only shared label space.  Each RGB cluster may be
    mapped to one IR cluster; RGB clusters that do not pass the graph check are
    filtered with label ``-1``.  This avoids the failed v14-union behaviour where
    unmerged RGB-only clusters created a much larger and weaker label space.
    """
    if rgb_view2 is None:
        rgb_view2 = rgb_features
    if ir_view2 is None:
        ir_view2 = ir_features

    rgb_labels_np = np.asarray(list(rgb_labels), dtype=np.int64)
    ir_labels_np = np.asarray(list(ir_labels), dtype=np.int64)
    rgb_keys = _valid_labels(rgb_labels_np)
    ir_keys = _valid_labels(ir_labels_np)
    if len(rgb_keys) == 0 or len(ir_keys) == 0:
        new_rgb = np.full_like(rgb_labels_np, -1)
        new_ir = ir_labels_np.copy()
        metrics = {
            'state': 'empty',
            'rgb_clusters': float(len(rgb_keys)),
            'ir_clusters': float(len(ir_keys)),
            'accepted_pairs': 0.0,
            'anchor_clusters': float(len(ir_keys)),
            'unified_clusters': float(len(ir_keys)),
            'cluster_coverage_rgb': 0.0,
            'cluster_coverage_ir': 0.0,
            'sample_coverage_rgb': 0.0,
            'sample_coverage_ir': 0.0,
            'neighbor_changed_top1': 0.0,
            'mean_neighbor_score': 0.0,
        }
        return new_rgb, new_ir, metrics

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

    rgb_to_ir_cluster, rgb_support = _aggregate_cluster_distance(
        final_rgb_ir, rgb_labels_np, ir_labels_np, rgb_keys, ir_keys,
        sample_topk, min_pairs)
    ir_to_rgb_cluster, ir_support = _aggregate_cluster_distance(
        final_ir_rgb, ir_labels_np, rgb_labels_np, ir_keys, rgb_keys,
        sample_topk, min_pairs)
    del final_rgb_ir, final_ir_rgb

    neighbor_scores, direct_scores, propagated_scores = _diffuse_cluster_affinity(
        rgb_to_ir_cluster, ir_to_rgb_cluster,
        rgb_cluster_topk, ir_cluster_topk, max_distance,
        neighborhood_blend, neighborhood_temperature)
    rgb_top = [_top_indices(rgb_to_ir_cluster[row], rgb_cluster_topk)
               for row in range(len(rgb_keys))]
    rgb_neighbor_top = [_top_score_indices(neighbor_scores[row], rgb_cluster_topk)
                        for row in range(len(rgb_keys))]
    ir_top = [_top_indices(ir_to_rgb_cluster[row], ir_cluster_topk)
              for row in range(len(ir_keys))]
    ir_top_sets = [set(row.tolist()) for row in ir_top]

    merges: List[MSPLFMerge] = []
    rgb_to_ir_label: Dict[int, int] = {}
    reciprocal_count = 0
    neighbor_changed_top1 = 0
    propagated_selected = 0
    selected_neighbor_scores = []
    selected_direct_scores = []
    for rgb_local, candidates in enumerate(rgb_top):
        best_merge = None
        best_score = -float('inf')
        best_neighbor_score = 0.0
        best_direct_score = 0.0
        candidate_ids = sorted(set(candidates.tolist()).union(
            set(rgb_neighbor_top[rgb_local].tolist())))
        for ir_local in candidate_ids:
            rgb_dist = float(rgb_to_ir_cluster[rgb_local, ir_local])
            ir_dist = float(ir_to_rgb_cluster[ir_local, rgb_local])
            if not np.isfinite(rgb_dist):
                continue
            if np.isfinite(ir_dist):
                avg_dist = 0.5 * (rgb_dist + ir_dist)
            else:
                avg_dist = rgb_dist
            rgb_margin = _margin_for_pair(rgb_to_ir_cluster[rgb_local], ir_local)
            reciprocal = rgb_local in ir_top_sets[ir_local]
            # For IR-anchor relabeling, use RGB-side confidence as the primary
            # gate.  Requiring mutual top-k caused v14 to cover only about a
            # quarter of RGB samples and broke Stage-2 supervision.
            margin = rgb_margin if np.isfinite(rgb_margin) else 0.0
            if avg_dist > float(max_distance):
                continue
            if margin < float(min_margin):
                continue
            neighbor_score = float(neighbor_scores[rgb_local, ir_local])
            direct_score = float(direct_scores[rgb_local, ir_local])
            merge = MSPLFMerge(
                rgb_cluster=int(rgb_keys[rgb_local]),
                ir_cluster=int(ir_keys[ir_local]),
                distance=avg_dist,
                margin=margin,
                rgb_support=int(rgb_support[rgb_local, ir_local]),
                ir_support=int(ir_support[ir_local, rgb_local]),
            )
            # Prefer reciprocal candidates when distances are close, but do not
            # make reciprocity mandatory.
            merge_score = (
                max(neighbor_score, 1e-6)
                + (0.03 if reciprocal else 0.0)
                + 0.02 * np.clip(margin, 0.0, 0.20)
                - 0.05 * avg_dist
            )
            if best_merge is None or merge_score > best_score:
                best_merge = merge
                best_score = merge_score
                best_neighbor_score = neighbor_score
                best_direct_score = direct_score
        if best_merge is not None:
            merges.append(best_merge)
            rgb_to_ir_label[int(best_merge.rgb_cluster)] = int(best_merge.ir_cluster)
            best_ir_local = int(np.where(ir_keys == int(best_merge.ir_cluster))[0][0])
            if rgb_local in ir_top_sets[best_ir_local]:
                reciprocal_count += 1
            direct_best = _top_indices(rgb_to_ir_cluster[rgb_local], 1)
            if direct_best.size and int(direct_best[0]) != best_ir_local:
                neighbor_changed_top1 += 1
            if float(propagated_scores[rgb_local, best_ir_local]) > best_direct_score:
                propagated_selected += 1
            selected_neighbor_scores.append(best_neighbor_score)
            selected_direct_scores.append(best_direct_score)

    new_rgb = np.array([
        rgb_to_ir_label.get(int(label), -1) for label in rgb_labels_np.tolist()
    ], dtype=np.int64)
    new_ir = ir_labels_np.copy()

    merged_rgb_labels = set(int(m.rgb_cluster) for m in merges)
    merged_ir_labels = set(int(m.ir_cluster) for m in merges)
    valid_rgb_mask = rgb_labels_np != -1
    valid_ir_mask = ir_labels_np != -1
    rgb_sample_cov = np.isin(rgb_labels_np, list(merged_rgb_labels))[valid_rgb_mask].mean()
    ir_sample_cov = np.isin(ir_labels_np, list(merged_ir_labels))[valid_ir_mask].mean()
    distances = np.array([m.distance for m in merges], dtype=np.float32)
    margins = np.array([m.margin for m in merges], dtype=np.float32)
    neighbor_score_values = np.array(selected_neighbor_scores, dtype=np.float32)
    direct_score_values = np.array(selected_direct_scores, dtype=np.float32)

    metrics = {
        'state': 'active',
        'rgb_clusters': float(len(rgb_keys)),
        'ir_clusters': float(len(ir_keys)),
        'accepted_pairs': float(len(merges)),
        'anchor_clusters': float(len(ir_keys)),
        'unified_clusters': float(len(ir_keys)),
        'cluster_coverage_rgb': float(len(merged_rgb_labels)) / float(max(1, len(rgb_keys))),
        'cluster_coverage_ir': float(len(merged_ir_labels)) / float(max(1, len(ir_keys))),
        'sample_coverage_rgb': float(rgb_sample_cov) if valid_rgb_mask.any() else 0.0,
        'sample_coverage_ir': float(ir_sample_cov) if valid_ir_mask.any() else 0.0,
        'reciprocal_rate': float(reciprocal_count) / float(max(1, len(merges))),
        'filtered_rgb_samples': float((new_rgb == -1).sum()),
        'mean_pair_distance': float(distances.mean()) if distances.size else 0.0,
        'mean_pair_margin': float(margins.mean()) if margins.size else 0.0,
        'sample_topk': float(sample_topk),
        'rgb_cluster_topk': float(rgb_cluster_topk),
        'ir_cluster_topk': float(ir_cluster_topk),
        'min_pairs': float(min_pairs),
        'max_distance': float(max_distance),
        'min_margin': float(min_margin),
        'neighborhood_blend': float(neighborhood_blend),
        'neighborhood_temperature': float(neighborhood_temperature),
        'neighbor_changed_top1': (
            float(neighbor_changed_top1) / float(max(1, len(merges)))),
        'propagated_selected': (
            float(propagated_selected) / float(max(1, len(merges)))),
        'mean_neighbor_score': (
            float(neighbor_score_values.mean()) if neighbor_score_values.size else 0.0),
        'mean_direct_score': (
            float(direct_score_values.mean()) if direct_score_values.size else 0.0),
        'exact_nodes': float(rgb_features.size(0) + ir_features.size(0)),
    }
    return new_rgb, new_ir, metrics
