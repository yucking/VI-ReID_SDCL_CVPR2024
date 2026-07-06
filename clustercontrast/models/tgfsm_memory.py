"""Topology-gated factorized shared memory for RGB-IR SDCL Stage-2."""

from __future__ import absolute_import

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def _cluster_prototypes(labels, features):
    labels = torch.as_tensor(labels, dtype=torch.long, device=features.device)
    valid = labels.ge(0)
    if not bool(valid.any()):
        return features.new_empty((0, features.size(1))), labels.new_empty((0,))
    cluster_ids, inverse = torch.unique(labels[valid], sorted=True, return_inverse=True)
    samples = F.normalize(features[valid], dim=1)
    sums = features.new_zeros((cluster_ids.numel(), features.size(1)))
    sums.index_add_(0, inverse, samples)
    counts = features.new_zeros(cluster_ids.numel())
    counts.index_add_(0, inverse, torch.ones_like(inverse, dtype=features.dtype))
    return F.normalize(sums / counts.clamp_min(1.0).unsqueeze(1), dim=1), cluster_ids


def _sinkhorn_transport(score, temperature, dustbin_score, dustbin_mass, iterations):
    num_rgb, num_ir = score.shape
    if num_rgb == 0 or num_ir == 0:
        return score.new_zeros((num_rgb + 1, num_ir + 1))
    augmented = score.new_full((num_rgb + 1, num_ir + 1), float(dustbin_score))
    augmented[:num_rgb, :num_ir] = score
    augmented[-1, -1] = 0.0
    log_kernel = augmented / max(float(temperature), 1e-4)
    rgb_marginal = torch.cat((
        score.new_full((num_rgb,), 1.0 / float(num_rgb)),
        score.new_tensor([float(dustbin_mass)]),
    )).log()
    ir_marginal = torch.cat((
        score.new_full((num_ir,), 1.0 / float(num_ir)),
        score.new_tensor([float(dustbin_mass)]),
    )).log()
    log_u = torch.zeros_like(rgb_marginal)
    log_v = torch.zeros_like(ir_marginal)
    for _ in range(max(1, int(iterations))):
        log_u = rgb_marginal - torch.logsumexp(log_kernel + log_v.unsqueeze(0), dim=1)
        log_v = ir_marginal - torch.logsumexp(log_kernel + log_u.unsqueeze(1), dim=0)
    return torch.exp(log_kernel + log_u.unsqueeze(1) + log_v.unsqueeze(0))


def _mutual_topk(score, topk):
    num_rgb, num_ir = score.shape
    if num_rgb == 0 or num_ir == 0:
        return torch.zeros_like(score, dtype=torch.bool)
    rgb_mask = torch.zeros_like(score, dtype=torch.bool)
    ir_mask = torch.zeros_like(score, dtype=torch.bool)
    rgb_mask.scatter_(1, score.topk(min(max(1, int(topk)), num_ir), dim=1).indices, True)
    ir_mask.scatter_(0, score.topk(min(max(1, int(topk)), num_rgb), dim=0).indices, True)
    return rgb_mask & ir_mask


def _persistent_mask(edge_indices, rgb_prototypes, ir_prototypes, state, threshold):
    if edge_indices.numel() == 0 or state is None:
        return torch.zeros(edge_indices.size(0), dtype=torch.bool, device=rgb_prototypes.device)
    previous_rgb = state.get('rgb')
    previous_ir = state.get('ir')
    if previous_rgb is None or previous_ir is None or previous_rgb.numel() == 0:
        return torch.zeros(edge_indices.size(0), dtype=torch.bool, device=rgb_prototypes.device)
    previous_rgb = previous_rgb.to(rgb_prototypes.device)
    previous_ir = previous_ir.to(ir_prototypes.device)
    rgb_similarity = rgb_prototypes[edge_indices[:, 0]].mm(previous_rgb.t())
    ir_similarity = ir_prototypes[edge_indices[:, 1]].mm(previous_ir.t())
    return torch.minimum(rgb_similarity, ir_similarity).max(dim=1).values.ge(float(threshold))


def _bounded_edges(edge_indices, confidence, num_rgb, num_ir, max_nodes_per_modal):
    rgb_degree = [0] * num_rgb
    ir_degree = [0] * num_ir
    selected = []
    rejected = 0
    if edge_indices.numel() == 0:
        return selected, rejected
    for order in confidence.argsort(descending=True).cpu().tolist():
        rgb_node, ir_node = edge_indices[order].cpu().tolist()
        if (rgb_degree[rgb_node] >= max_nodes_per_modal or
                ir_degree[ir_node] >= max_nodes_per_modal):
            rejected += 1
            continue
        rgb_degree[rgb_node] += 1
        ir_degree[ir_node] += 1
        selected.append((int(rgb_node), int(ir_node)))
    return selected, rejected


def build_topology_trust_masks(
        rgb_local_labels,
        ir_local_labels,
        rgb_shared_labels,
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
    """Return graph-certified samples without changing the shared label space."""
    rgb_proto, rgb_ids = _cluster_prototypes(rgb_local_labels, rgb_features)
    ir_proto, ir_ids = _cluster_prototypes(ir_local_labels, ir_features)
    rgb_proto_aux, _ = _cluster_prototypes(rgb_local_labels, rgb_features_aux)
    ir_proto_aux, _ = _cluster_prototypes(ir_local_labels, ir_features_aux)
    num_rgb = rgb_proto.size(0)
    num_ir = ir_proto.size(0)
    rgb_trust = np.zeros(len(rgb_local_labels), dtype=np.float32)
    ir_trust = np.zeros(len(ir_local_labels), dtype=np.float32)
    diagnostics = {
        'rgb_clusters': int(num_rgb), 'ir_clusters': int(num_ir),
        'candidates': 0, 'persistent': 0, 'accepted': 0,
        'rgb_trust': 0.0, 'ir_trust': 0.0, 'entropy': 0.0,
        'mean_score': 0.0, 'mean_confidence': 0.0, 'capacity_rejected': 0,
    }
    if num_rgb == 0 or num_ir == 0:
        return rgb_trust, ir_trust, None, diagnostics

    primary_score = rgb_proto.mm(ir_proto.t())
    auxiliary_score = rgb_proto_aux.mm(ir_proto_aux.t())
    score = 0.5 * (primary_score + auxiliary_score)
    transport = _sinkhorn_transport(score, temperature, dustbin_score, dustbin_mass, sinkhorn_iterations)
    real_transport = transport[:num_rgb, :num_ir]
    row_probability = real_transport / real_transport.sum(dim=1, keepdim=True).clamp_min(1e-12)
    col_probability = real_transport / real_transport.sum(dim=0, keepdim=True).clamp_min(1e-12)
    confidence = torch.sqrt(row_probability * col_probability)
    dustbin_mask = ((real_transport > transport[:num_rgb, num_ir].unsqueeze(1)) &
                    (real_transport > transport[num_rgb, :num_ir].unsqueeze(0)))
    valid_mask = (_mutual_topk(score, topk) & dustbin_mask &
                  score.ge(float(min_similarity)) &
                  torch.minimum(primary_score, auxiliary_score).ge(float(min_view_similarity)) &
                  confidence.ge(float(min_transport_confidence)))
    candidate_indices = valid_mask.nonzero(as_tuple=False)
    persistent = _persistent_mask(candidate_indices, rgb_proto, ir_proto, previous_state, persist_similarity)
    persistent_indices = candidate_indices[persistent]
    selected_edges, capacity_rejected = _bounded_edges(
        persistent_indices,
        confidence[persistent_indices[:, 0], persistent_indices[:, 1]] if persistent_indices.numel() else confidence.new_empty((0,)),
        num_rgb, num_ir, int(max_nodes_per_modal),
    )
    next_state = None
    if candidate_indices.numel() > 0:
        next_state = {
            'rgb': rgb_proto[candidate_indices[:, 0]].detach().cpu(),
            'ir': ir_proto[candidate_indices[:, 1]].detach().cpu(),
        }

    rgb_local_labels = np.asarray(rgb_local_labels)
    ir_local_labels = np.asarray(ir_local_labels)
    rgb_shared_labels = np.asarray(rgb_shared_labels)
    for rgb_node, ir_node in selected_edges:
        rgb_cluster = int(rgb_ids[rgb_node].item())
        ir_cluster = int(ir_ids[ir_node].item())
        rgb_trust[(rgb_local_labels == rgb_cluster) & (rgb_shared_labels == ir_cluster)] = 1.0
        ir_trust[ir_local_labels == ir_cluster] = 1.0

    entropy = -(row_probability * row_probability.clamp_min(1e-12).log()).sum(dim=1)
    diagnostics.update({
        'candidates': int(candidate_indices.size(0)),
        'persistent': int(persistent_indices.size(0)),
        'accepted': int(len(selected_edges)),
        'rgb_trust': float(rgb_trust.mean()),
        'ir_trust': float(ir_trust.mean()),
        'entropy': float((entropy / max(math.log(float(max(num_ir, 2))), 1.0)).mean().item()),
        'mean_score': float(score[candidate_indices[:, 0], candidate_indices[:, 1]].mean().item()) if candidate_indices.numel() else 0.0,
        'mean_confidence': float(confidence[candidate_indices[:, 0], candidate_indices[:, 1]].mean().item()) if candidate_indices.numel() else 0.0,
        'capacity_rejected': int(capacity_rejected),
    })
    return rgb_trust, ir_trust, next_state, diagnostics


def _class_prototypes(labels, features, num_classes, fallback):
    labels = torch.as_tensor(labels, dtype=torch.long, device=features.device)
    samples = F.normalize(features, dim=1)
    valid = labels.ge(0) & labels.lt(num_classes)
    result = F.normalize(fallback.clone(), dim=1)
    if not bool(valid.any()):
        return result
    sums = features.new_zeros((num_classes, features.size(1)))
    counts = features.new_zeros(num_classes)
    sums.index_add_(0, labels[valid], samples[valid])
    counts.index_add_(0, labels[valid], torch.ones_like(labels[valid], dtype=features.dtype))
    active = counts.gt(0)
    result[active] = F.normalize(sums[active] / counts[active].unsqueeze(1), dim=1)
    return result


def _initial_proxies(labels, features, num_classes, num_proxies, fallback):
    labels = torch.as_tensor(labels, dtype=torch.long, device=features.device)
    samples = F.normalize(features, dim=1)
    class_centers = _class_prototypes(labels, samples, num_classes, fallback)
    proxies = class_centers.unsqueeze(1).repeat(1, num_proxies, 1)
    if num_proxies == 1:
        return proxies
    for class_index in range(num_classes):
        class_samples = samples[labels.eq(class_index)]
        if class_samples.size(0) < num_proxies:
            continue
        selected = [int(class_samples.mm(class_centers[class_index].unsqueeze(1)).argmax().item())]
        for _ in range(1, num_proxies):
            selected_features = class_samples[selected]
            nearest_similarity = class_samples.mm(selected_features.t()).max(dim=1).values
            selected.append(int(nearest_similarity.argmin().item()))
        proxies[class_index] = class_samples[selected]
    return F.normalize(proxies, dim=2)


class TopologyGatedFactorizedMemory(nn.Module):
    """Class-count preserving memory with modality-private and gated consensus proxies."""
    def __init__(self, num_features, num_classes, num_proxies=2, temp=0.05,
                 momentum=0.1, consensus_momentum=0.9, consensus_weight=0.15):
        super(TopologyGatedFactorizedMemory, self).__init__()
        self.num_features = int(num_features)
        self.num_classes = int(num_classes)
        self.num_proxies = int(num_proxies)
        self.temp = float(temp)
        self.momentum = float(momentum)
        self.consensus_momentum = float(consensus_momentum)
        self.consensus_weight = float(consensus_weight)
        self.register_buffer('rgb_proxies', torch.zeros(num_classes, num_proxies, num_features))
        self.register_buffer('ir_proxies', torch.zeros(num_classes, num_proxies, num_features))
        self.register_buffer('consensus', torch.zeros(num_classes, num_features))

    @torch.no_grad()
    def initialize(self, rgb_features, rgb_labels, ir_features, ir_labels, shared_features):
        shared_features = F.normalize(shared_features, dim=1)
        if shared_features.size(0) != self.num_classes:
            raise ValueError('TG-FSM expected {} shared prototypes, got {}.'.format(
                self.num_classes, shared_features.size(0)
            ))
        ir_centers = _class_prototypes(ir_labels, ir_features, self.num_classes, shared_features)
        self.consensus.copy_(ir_centers)
        self.rgb_proxies.copy_(_initial_proxies(
            rgb_labels, rgb_features, self.num_classes, self.num_proxies, ir_centers
        ))
        self.ir_proxies.copy_(_initial_proxies(
            ir_labels, ir_features, self.num_classes, self.num_proxies, ir_centers
        ))

    @torch.no_grad()
    def _update_private(self, inputs, targets, modality):
        proxies = self.rgb_proxies if modality == 0 else self.ir_proxies
        selected = proxies[targets]
        slots = torch.einsum('bd,bkd->bk', inputs, selected).argmax(dim=1)
        flat_ids = targets * self.num_proxies + slots
        unique_ids, inverse = torch.unique(flat_ids, sorted=False, return_inverse=True)
        sums = inputs.new_zeros((unique_ids.numel(), inputs.size(1)))
        sums.index_add_(0, inverse, inputs)
        counts = inputs.new_zeros(unique_ids.numel())
        counts.index_add_(0, inverse, torch.ones_like(inverse, dtype=inputs.dtype))
        means = F.normalize(sums / counts.clamp_min(1.0).unsqueeze(1), dim=1)
        flat_proxies = proxies.view(self.num_classes * self.num_proxies, self.num_features)
        flat_proxies[unique_ids] = F.normalize(
            self.momentum * flat_proxies[unique_ids] + (1.0 - self.momentum) * means,
            dim=1,
        )

    @torch.no_grad()
    def _update_consensus(self, inputs, targets, trust):
        active = trust.gt(0.5)
        if not bool(active.any()):
            return
        active_inputs = inputs[active]
        active_targets = targets[active]
        unique_ids, inverse = torch.unique(active_targets, sorted=False, return_inverse=True)
        sums = active_inputs.new_zeros((unique_ids.numel(), active_inputs.size(1)))
        sums.index_add_(0, inverse, active_inputs)
        counts = active_inputs.new_zeros(unique_ids.numel())
        counts.index_add_(0, inverse, torch.ones_like(inverse, dtype=active_inputs.dtype))
        means = F.normalize(sums / counts.clamp_min(1.0).unsqueeze(1), dim=1)
        self.consensus[unique_ids] = F.normalize(
            self.consensus_momentum * self.consensus[unique_ids]
            + (1.0 - self.consensus_momentum) * means,
            dim=1,
        )

    def forward(self, inputs, targets, modality, ca=None):
        inputs = F.normalize(inputs, dim=1)
        targets = targets.long()
        if ca is None:
            loss_weight = torch.ones(inputs.size(0), device=inputs.device, dtype=inputs.dtype)
            trust = torch.zeros_like(loss_weight)
        elif ca.dim() == 1:
            loss_weight = ca.to(inputs.device, dtype=inputs.dtype)
            trust = torch.zeros_like(loss_weight)
        else:
            loss_weight = ca[:, 0].to(inputs.device, dtype=inputs.dtype)
            trust = ca[:, 1].to(inputs.device, dtype=inputs.dtype)

        private = (self.rgb_proxies if modality == 0 else self.ir_proxies).detach().clone()
        consensus = self.consensus.detach().clone()
        private_logits = torch.einsum('bd,ckd->bck', inputs, private) / self.temp
        private_logits = torch.logsumexp(private_logits, dim=2) - math.log(float(self.num_proxies))
        consensus_logits = inputs.mm(consensus.t()) / self.temp
        gate = (trust * self.consensus_weight).clamp(0.0, self.consensus_weight).unsqueeze(1)
        logits = private_logits + gate * (consensus_logits - private_logits)
        loss = (F.cross_entropy(logits, targets, reduction='none') * loss_weight).mean()

        with torch.no_grad():
            self._update_private(inputs.detach(), targets, modality)
            self._update_consensus(inputs.detach(), targets, trust)
        return loss


class TopologyGatedMemoryView(nn.Module):
    """Trainer-compatible modal view over one factorized memory bank."""
    def __init__(self, bank, modality):
        super(TopologyGatedMemoryView, self).__init__()
        self.bank = bank
        self.modality = int(modality)

    def forward(self, inputs, targets, ca=None, training_momentum=None):
        del training_momentum
        return self.bank(inputs, targets, self.modality, ca=ca)
