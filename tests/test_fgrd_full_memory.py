"""Smoke tests for FGRD-FULL."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from clustercontrast.models.fgrd_full_memory import (
    build_fgrd_full_teacher,
    fgrd_full_ranking_loss,
    fuse_metric_features,
)


def _assert_metric_fusion_shape():
    torch.manual_seed(1)
    global_features = torch.randn(5, 8)
    local_features = torch.randn(5, 8)
    fused = fuse_metric_features(global_features, local_features, 0.25)
    assert fused.shape == (5, 16)
    assert torch.isfinite(fused).all()


def _assert_full_teacher_and_loss():
    torch.manual_seed(7)
    rgb = torch.randn(18, 12)
    ir = torch.randn(16, 12)
    rgb_local = torch.randn(18, 12)
    ir_local = torch.randn(16, 12)

    ir[:4] = rgb[:4] + 0.02 * torch.randn(4, 12)
    ir_local[:4] = rgb_local[:4] + 0.02 * torch.randn(4, 12)

    rgb_to_ir, ir_to_rgb, metrics = build_fgrd_full_teacher(
        rgb, ir,
        rgb_view2=rgb_local,
        ir_view2=ir_local,
        global_weight=0.25,
        positive_count=4,
        negative_count=4,
        negative_start=8,
        rerank_k1=6,
        rerank_k2=3,
        rerank_lambda=0.10,
        csls_neighbors=3,
        csls_blend=0.75,
        confidence_floor=0.0,
        entropy_ceiling=1.0,
        seed=1,
    )
    assert rgb_to_ir.candidate_ids.shape == (18, 8)
    assert ir_to_rgb.candidate_ids.shape == (16, 8)
    assert metrics["coverage_rgb"] > 0.0
    assert metrics["coverage_ir"] > 0.0
    assert metrics["exact_nodes_rgb_to_ir"] == 34
    assert metrics["candidate_count"] == 8
    assert torch.isfinite(rgb_to_ir.teacher_probs).all()
    assert rgb_to_ir.positive_mask[:, :4].all()
    assert rgb_to_ir.negative_mask[:, 4:].all()

    student = rgb.clone().requires_grad_(True)
    loss, stats = fgrd_full_ranking_loss(
        student, torch.arange(18), ir, rgb_to_ir,
        temperature=0.05, margin=0.20, pairwise_weight=0.50)
    assert torch.isfinite(loss), loss
    assert stats["active_rows"] > 0.0
    assert stats["listwise"] > 0.0
    loss.backward()
    assert student.grad is not None and torch.isfinite(student.grad).all()


if __name__ == "__main__":
    _assert_metric_fusion_shape()
    _assert_full_teacher_and_loss()
    print("FGRD-FULL smoke test passed")
