from __future__ import absolute_import

from abc import ABC

import torch
import torch.nn.functional as F
from torch import nn

from .cm import cm, cm_hard


class CETMMemory(nn.Module, ABC):
    """Cluster memory with conservative set-valued RGB transport positives.

    The prototype tensor can be shared with a strict ``ClusterMemory``.  The
    hard pseudo label remains the only target used by the memory update; the
    positive mask affects only the contrastive numerator.
    """

    def __init__(
        self,
        num_features,
        num_samples,
        positive_mask,
        temp=0.05,
        momentum=0.2,
        use_hard=False,
        features=None,
    ):
        super(CETMMemory, self).__init__()
        if positive_mask.shape != (num_samples, num_samples):
            raise ValueError(
                'positive_mask must have shape ({}, {}), got {}'.format(
                    num_samples, num_samples, tuple(positive_mask.shape)
                )
            )

        self.num_features = num_features
        self.num_samples = num_samples
        self.momentum = momentum
        self.temp = temp
        self.use_hard = use_hard

        if features is None:
            features = torch.zeros(num_samples, num_features)
        if tuple(features.shape) != (num_samples, num_features):
            raise ValueError(
                'features must have shape ({}, {}), got {}'.format(
                    num_samples, num_features, tuple(features.shape)
                )
            )

        # Keeping this reference shared preserves SDCL's joint RGB-IR memory.
        self.register_buffer('features', features)
        self.register_buffer(
            'positive_mask', positive_mask.to(device=features.device, dtype=torch.bool)
        )

    def forward(self, inputs, targets, ca=None, training_momentum=None):
        inputs = F.normalize(inputs, dim=1).to(self.features.device)
        targets = targets.long().to(self.features.device)
        momentum = self.momentum if training_momentum is None else training_momentum

        if self.use_hard:
            logits = cm_hard(inputs, targets, self.features, momentum)
        else:
            logits = cm(inputs, targets, self.features, momentum)
        logits = logits / self.temp

        positive_rows = self.positive_mask.index_select(0, targets)
        positive_logits = logits.masked_fill(~positive_rows, float('-inf'))
        loss_per_sample = torch.logsumexp(logits, dim=1) - torch.logsumexp(
            positive_logits, dim=1
        )

        if ca is None:
            return loss_per_sample.mean()
        return (loss_per_sample * ca.to(self.features.device)).mean()
