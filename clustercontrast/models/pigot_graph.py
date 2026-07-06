"""Persistent partial-transport identity graph for RGB-IR Stage-2 labels."""

from __future__ import absolute_import

import math

import numpy as np
import torch
import torch.nn.functional as F


def _cluster_prototypes(labels, features):
    """Return normalized prototypes, DBSCAN ids, and cluster sizes."""
    labels = torch.as_tensor(labels, dtype=torch.long, device=features.device)
    valid_mask = labels.ge(0)
    if not bool(valid_mask.any()):
        empty_proto = features.new_empty((0, features.size(1)))
        empty_ids = labels.new_empty((0,))
        return empty_proto, empty_ids, features.new_empty((0,))

    cluster_ids, inverse = torch.unique(labels[valid_mask], sorted=True, return_inverse=True)
    cluster_features = F.normalize(features[valid_mask], dim=1)
    prototypes = features.new_zeros((cluster_ids.numel(), features.size(1)))
    prototypes.index_add_(0, inverse, cluster_features)
    sizes = features.new_zeros(cluster_ids.numel())
    sizes.index_add_(0, inverse, torch.ones_like(inverse, dtype=features.dtype))
    prototypes = F.normalize(prototypes / sizes.clamp_min(1.0).unsqueeze(1), dim=1)
    return prototypes, cluster_ids, sizes


def _sinkhorn_partial_transport(score, temperature, dustbin_score, dustbin_mass, iterations):
    """Balanced Sinkhorn transport with one dustbin node on each modality."""
    num_rgb, num_ir = score.shape
    if num_rgb == 0 or num_ir == 0:
        return score.new_zeros((num_rgb + 1, num_ir + 1))

    transport_score = score.new_full((num_rgb + 1, num_ir + 1), dustbin_score)
    transport_score[:num_rgb, :num_ir] = score
    transport_score[-1, -1] = 0.0
    log_kernel = transport_score / max(float(temperature), 1e-4)

    rgb_marginal = torch.cat((
        score.new_full((num_rgb,), 1.0 / float(num_rgb)),
        score.new_tensor([float(dustbin_mass)]),
    ))
    ir_marginal = torch.cat((
        score.new_full((num_ir,), 1.0 / float(num_ir)),
        score.new_tensor([float(dustbin_mass)]),
    ))
    log_rgb_marginal = rgb_marginal.log()
    log_ir_marginal = ir_marginal.log()
    log_u = torch.zeros_like(log_rgb_marginal)
    log_v = torch.zeros_like(log_ir_marginal)
    for _ in range(max(1, int(iterations))):
        log_u = log_rgb_marginal - torch.logsumexp(log_kernel + log_v.unsqueeze(0), dim=1)
        log_v = log_ir_marginal - torch.logsumexp(log_kernel + log_u.unsqueeze(1), dim=0)
    return torch.exp(log_kernel + log_u.unsqueeze(1) + log_v.unsqueeze(0))


def _mutual_topk_mask(score, topk):
    num_rgb, num_ir = score.shape
    mask = torch.zeros_like(score, dtype=torch.bool)
    if num_rgb == 0 or num_ir == 0:
        return mask
    rgb_topk = min(max(1, int(topk)), num_ir)
    ir_topk = min(max(1, int(topk)), num_rgb)
    rgb_indices = score.topk(rgb_topk, dim=1).indices
    ir_indices = score.topk(ir_topk, dim=0).indices
    rgb_mask = torch.zeros_like(mask)
    ir_mask = torch.zeros_like(mask)
    rgb_mask.scatter_(1, rgb_indices, True)
    ir_mask.scatter_(0, ir_indices, True)
    return rgb_mask & ir_mask


def _persistent_edge_mask(edge_indices, rgb_prototypes, ir_prototypes, previous_state, threshold):
    if edge_indices.numel() == 0 or previous_state is None:
        return torch.zeros(edge_indices.size(0), dtype=torch.bool, device=rgb_prototypes.device)

    previous_rgb = previous_state.get('candidate_rgb_prototypes')
    previous_ir = previous_state.get('candidate_ir_prototypes')
    if previous_rgb is None or previous_ir is None or previous_rgb.numel() == 0:
        return torch.zeros(edge_indices.size(0), dtype=torch.bool, device=rgb_prototypes.device)

    previous_rgb = previous_rgb.to(rgb_prototypes.device)
    previous_ir = previous_ir.to(ir_prototypes.device)
    rgb_similarity = rgb_prototypes[edge_indices[:, 0]].mm(previous_rgb.t())
    ir_similarity = ir_prototypes[edge_indices[:, 1]].mm(previous_ir.t())
    pair_similarity = torch.minimum(rgb_similarity, ir_similarity).max(dim=1).values
    return pair_similarity.ge(float(threshold))


class _UnionFind(object):
    def __init__(self, num_rgb, num_ir):
        total = num_rgb + num_ir
        self.parent = list(range(total))
        self.rgb_count = [1] * num_rgb + [0] * num_ir
        self.ir_count = [0] * num_rgb + [1] * num_ir

    def find(self, node):
        while self.parent[node] != node:
            self.parent[node] = self.parent[self.parent[node]]
            node = self.parent[node]
        return node

    def union(self, rgb_node, ir_node, max_nodes_per_modal):
        rgb_root = self.find(rgb_node)
        ir_root = self.find(ir_node)
        if rgb_root == ir_root:
            return True
        if (self.rgb_count[rgb_root] + self.rgb_count[ir_root] > max_nodes_per_modal or
                self.ir_count[rgb_root] + self.ir_count[ir_root] > max_nodes_per_modal):
            return False
        self.parent[ir_root] = rgb_root
        self.rgb_count[rgb_root] += self.rgb_count[ir_root]
        self.ir_count[rgb_root] += self.ir_count[ir_root]
        return True


def _labels_from_components(labels, cluster_ids, node_labels):
    lookup = {int(cluster_id): int(node_labels[node]) for node, cluster_id in enumerate(cluster_ids.cpu().tolist())}
    output = np.full(len(labels), -1, dtype=np.int64)
    for index, cluster_id in enumerate(labels):
        if int(cluster_id) >= 0:
            output[index] = lookup[int(cluster_id)]
    return output


def build_pigot_identity_graph(
        rgb_labels,
        ir_labels,
        rgb_features,
        ir_features,
        rgb_features_aux,
        ir_features_aux,
        previous_state=None,
        topk=2,
        temperature=0.07,
        dustbin_score=0.35,
        dustbin_mass=0.35,
        sinkhorn_iterations=30,
        min_similarity=0.35,
        min_view_similarity=0.30,
        min_transport_confidence=0.05,
        persist_similarity=0.92,
        max_nodes_per_modal=2):
    """Build conservative shared IDs without forcing every RGB cluster into IR."""
    rgb_prototypes, rgb_cluster_ids, _ = _cluster_prototypes(rgb_labels, rgb_features)
    ir_prototypes, ir_cluster_ids, _ = _cluster_prototypes(ir_labels, ir_features)
    rgb_prototypes_aux, _, _ = _cluster_prototypes(rgb_labels, rgb_features_aux)
    ir_prototypes_aux, _, _ = _cluster_prototypes(ir_labels, ir_features_aux)
    num_rgb = rgb_prototypes.size(0)
    num_ir = ir_prototypes.size(0)

    if num_rgb == 0 or num_ir == 0:
        rgb_private = np.full(len(rgb_labels), -1, dtype=np.int64)
        ir_private = np.full(len(ir_labels), -1, dtype=np.int64)
        diagnostics = {
            'rgb_clusters': int(num_rgb), 'ir_clusters': int(num_ir),
            'candidate_edges': 0, 'persistent_edges': 0, 'accepted_edges': 0,
            'rgb_coverage': 0.0, 'ir_coverage': 0.0, 'shared_ids': int(num_rgb + num_ir),
            'mean_transport_entropy': 0.0, 'capacity_rejected': 0,
        }
        return rgb_private, ir_private, int(num_rgb + num_ir), None, diagnostics

    primary_score = rgb_prototypes.mm(ir_prototypes.t())
    auxiliary_score = rgb_prototypes_aux.mm(ir_prototypes_aux.t())
    score = 0.5 * (primary_score + auxiliary_score)
    transport = _sinkhorn_partial_transport(
        score, temperature, dustbin_score, dustbin_mass, sinkhorn_iterations)
    real_transport = transport[:num_rgb, :num_ir]
    row_probability = real_transport / real_transport.sum(dim=1, keepdim=True).clamp_min(1e-12)
    col_probability = real_transport / real_transport.sum(dim=0, keepdim=True).clamp_min(1e-12)
    transport_confidence = torch.sqrt(row_probability * col_probability)
    mutual_mask = _mutual_topk_mask(score, topk)
    dustbin_mask = ((real_transport > transport[:num_rgb, num_ir].unsqueeze(1)) &
                    (real_transport > transport[num_rgb, :num_ir].unsqueeze(0)))
    view_mask = torch.minimum(primary_score, auxiliary_score).ge(float(min_view_similarity))
    base_mask = (mutual_mask & dustbin_mask & view_mask &
                 score.ge(float(min_similarity)) &
                 transport_confidence.ge(float(min_transport_confidence)))
    candidate_indices = base_mask.nonzero(as_tuple=False)
    persistent_mask = _persistent_edge_mask(
        candidate_indices, rgb_prototypes, ir_prototypes, previous_state, persist_similarity)
    accepted_indices = candidate_indices[persistent_mask]

    # Keep the next graph update honest: persistence is measured against all valid candidates,
    # not only the edges that survived the component-capacity check.
    next_state = None
    if candidate_indices.numel() > 0:
        next_state = {
            'candidate_rgb_prototypes': rgb_prototypes[candidate_indices[:, 0]].detach().cpu(),
            'candidate_ir_prototypes': ir_prototypes[candidate_indices[:, 1]].detach().cpu(),
        }

    union_find = _UnionFind(num_rgb, num_ir)
    capacity_rejected = 0
    kept_edges = []
    if accepted_indices.numel() > 0:
        edge_confidence = transport_confidence[accepted_indices[:, 0], accepted_indices[:, 1]]
        edge_order = edge_confidence.argsort(descending=True)
        for edge_index in edge_order.cpu().tolist():
            rgb_node, ir_node = accepted_indices[edge_index].cpu().tolist()
            if union_find.union(int(rgb_node), int(num_rgb + ir_node), int(max_nodes_per_modal)):
                kept_edges.append((int(rgb_node), int(ir_node)))
            else:
                capacity_rejected += 1

    root_to_shared = {}
    node_to_shared = []
    for node in range(num_rgb + num_ir):
        root = union_find.find(node)
        if root not in root_to_shared:
            root_to_shared[root] = len(root_to_shared)
        node_to_shared.append(root_to_shared[root])
    rgb_shared = _labels_from_components(rgb_labels, rgb_cluster_ids, node_to_shared[:num_rgb])
    ir_shared = _labels_from_components(ir_labels, ir_cluster_ids, node_to_shared[num_rgb:])

    linked_rgb = {edge[0] for edge in kept_edges}
    linked_ir = {edge[1] for edge in kept_edges}
    entropy = -(row_probability * row_probability.clamp_min(1e-12).log()).sum(dim=1)
    entropy = entropy / max(math.log(float(max(num_ir, 2))), 1.0)
    diagnostics = {
        'rgb_clusters': int(num_rgb),
        'ir_clusters': int(num_ir),
        'candidate_edges': int(candidate_indices.size(0)),
        'persistent_edges': int(accepted_indices.size(0)),
        'accepted_edges': int(len(kept_edges)),
        'rgb_coverage': float(len(linked_rgb)) / float(max(num_rgb, 1)),
        'ir_coverage': float(len(linked_ir)) / float(max(num_ir, 1)),
        'shared_ids': int(len(root_to_shared)),
        'mean_transport_entropy': float(entropy.mean().item()),
        'capacity_rejected': int(capacity_rejected),
        'mean_edge_score': float(score[candidate_indices[:, 0], candidate_indices[:, 1]].mean().item()) if candidate_indices.numel() else 0.0,
        'mean_edge_confidence': float(transport_confidence[candidate_indices[:, 0], candidate_indices[:, 1]].mean().item()) if candidate_indices.numel() else 0.0,
    }
    return rgb_shared, ir_shared, int(len(root_to_shared)), next_state, diagnostics
