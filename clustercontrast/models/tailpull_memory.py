from __future__ import absolute_import

from abc import ABC

import torch
from torch import nn
from torch.nn import functional as F

from clustercontrast.models.cm import cm, cm_hard


class TailPullMemory(nn.Module, ABC):
    """Cluster memory with a conservative tail prototype pull.

    The hard pseudo label remains the only class target and the memory update is
    identical to ClusterMemory. The extra term only pulls samples from loose
    pseudo classes back toward their assigned prototype.
    """

    def __init__(
        self,
        num_features,
        num_samples,
        temp=0.05,
        momentum=0.2,
        use_hard=False,
        class_tail_weight=None,
        pull_weight=0.08,
        pull_margin=0.65,
        pull_warmup=1.0,
    ):
        super(TailPullMemory, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples
        self.momentum = momentum
        self.temp = temp
        self.use_hard = use_hard
        self.pull_weight = float(pull_weight)
        self.pull_margin = float(pull_margin)
        self.pull_warmup = float(pull_warmup)

        self.register_buffer('features', torch.zeros(num_samples, num_features))
        if class_tail_weight is None:
            class_tail_weight = torch.ones(num_samples)
        class_tail_weight = class_tail_weight.float().view(-1)
        if class_tail_weight.numel() != num_samples:
            raise ValueError('class_tail_weight size must match num_samples')
        self.register_buffer('class_tail_weight', class_tail_weight)

    def forward(self, inputs, targets, ca=None, training_momentum=None):
        inputs = F.normalize(inputs, dim=1).cuda()
        momentum = self.momentum if training_momentum is None else training_momentum
        if self.use_hard:
            outputs = cm_hard(inputs, targets, self.features, momentum)
        else:
            outputs = cm(inputs, targets, self.features, momentum)

        outputs /= self.temp
        ce_loss = F.cross_entropy(outputs, targets, reduction='none')
        sample_weight = None if ca is None else ca.detach().float()
        if sample_weight is None:
            loss = ce_loss.mean()
        else:
            loss = (ce_loss * sample_weight).mean()

        if self.pull_weight <= 0.0 or self.pull_warmup <= 0.0:
            return loss

        with torch.no_grad():
            target_proto = self.features.index_select(0, targets).detach()
            class_weight = self.class_tail_weight.to(inputs.device).index_select(0, targets)

        own_sim = (inputs * target_proto).sum(dim=1)
        pull = F.relu(self.pull_margin - own_sim).pow(2)
        pull_scale = class_weight
        if sample_weight is not None:
            pull_scale = pull_scale * sample_weight.to(inputs.device)
        pull_denom = pull_scale.sum().clamp_min(1e-6)
        pull_loss = (pull * pull_scale).sum() / pull_denom
        return loss + self.pull_weight * self.pull_warmup * pull_loss
