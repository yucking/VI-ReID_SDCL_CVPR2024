"""Smoke tests for Full-Graph Ranking Distillation."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from clustercontrast.models.fgrd_memory import (
    build_fgrd_teacher,
    fgrd_ranking_loss,
    fuse_metric_features,
)


def _assert_metric_fusion_shape():
    torch.manual_seed(1)
    global_features = torch.randn(5, 8)
    local_features = torch.randn(5, 8)
    fused = fuse_metric_features(global_features, local_features, 0.25)
    assert fused.shape == (5, 16)
    assert torch.isfinite(fused).all()


def _assert_fgrd_teacher_and_loss():
    torch.manual_seed(7)
    rgb = torch.randn(18, 12)
    ir = torch.randn(16, 12)
    rgb_local = torch.randn(18, 12)
    ir_local = torch.randn(16, 12)
    rgb_cams = torch.tensor([0, 1, 3] * 6)
    ir_cams = torch.tensor([4, 6] * 8)

    # Strong cross-modal neighbours should survive exact reranking and become
    # positive ranking targets.
    ir[:4] = rgb[:4] + 0.02 * torch.randn(4, 12)
    ir_local[:4] = rgb_local[:4] + 0.02 * torch.randn(4, 12)

    rgb_to_ir, ir_to_rgb, metrics = build_fgrd_teacher(
        rgb, ir,
        rgb_view2=rgb_local,
        ir_view2=ir_local,
        rgb_cameras=rgb_cams,
        ir_cameras=ir_cams,
        global_weight=0.25,
        query_per_camera=4,
        gallery_per_camera=5,
        positive_count=2,
        hard_negative_count=4,
        mutual_topk=8,
        rerank_k1=6,
        rerank_k2=3,
        rerank_lambda=0.10,
        csls_neighbors=3,
        csls_blend=0.75,
        confidence_floor=0.0,
        entropy_ceiling=1.0,
        seed=1,
    )
    assert rgb_to_ir.candidate_ids.shape == (18, 6)
    assert ir_to_rgb.candidate_ids.shape == (16, 6)
    assert 0.0 < metrics["coverage_rgb"] <= 1.0
    assert 0.0 < metrics["coverage_ir"] <= 1.0
    assert metrics["exact_nodes_rgb_to_ir"] > 0
    assert metrics["candidate_count"] == 6
    assert torch.isfinite(rgb_to_ir.teacher_probs).all()
    assert rgb_to_ir.positive_mask.any()
    assert rgb_to_ir.negative_mask.any()

    student = rgb.clone().requires_grad_(True)
    loss, stats = fgrd_ranking_loss(student, torch.arange(18), ir, rgb_to_ir)
    assert torch.isfinite(loss), loss
    assert stats["active_rows"] > 0.0
    assert stats["listwise"] > 0.0
    loss.backward()
    assert student.grad is not None and torch.isfinite(student.grad).all()


if __name__ == "__main__":
    _assert_metric_fusion_shape()
    _assert_fgrd_teacher_and_loss()
    print("FGRD smoke test passed")
