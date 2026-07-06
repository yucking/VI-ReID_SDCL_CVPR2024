import importlib.util
from pathlib import Path
import sys
import types

import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[1]
_RERANK_PATH = _ROOT / 'clustercontrast' / 'utils' / 'rerank.py'
_RERANK_SPEC = importlib.util.spec_from_file_location(
    'clustercontrast.utils.rerank', _RERANK_PATH)
_RERANK_MODULE = importlib.util.module_from_spec(_RERANK_SPEC)
sys.modules.setdefault('clustercontrast', types.ModuleType('clustercontrast'))
sys.modules.setdefault('clustercontrast.utils', types.ModuleType('clustercontrast.utils'))
sys.modules[_RERANK_SPEC.name] = _RERANK_MODULE
_RERANK_SPEC.loader.exec_module(_RERANK_MODULE)

_MODULE_PATH = (Path(__file__).resolve().parents[1] /
                'clustercontrast' / 'models' / 'gprd_gap_memory.py')
_SPEC = importlib.util.spec_from_file_location(
    'gprd_gap_memory_under_test', _MODULE_PATH)
_gprd_gap_memory = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _gprd_gap_memory
_SPEC.loader.exec_module(_gprd_gap_memory)

GPRDRelation = _gprd_gap_memory.GPRDRelation
build_gprd_teacher = _gprd_gap_memory.build_gprd_teacher
merge_gprd_teacher_cache = _gprd_gap_memory.merge_gprd_teacher_cache
gprd_ranking_loss = _gprd_gap_memory.gprd_ranking_loss


def test_gprd_gap_teacher_keeps_stable_reciprocal_edges():
    torch.manual_seed(11)
    centers = F.normalize(torch.randn(4, 8), dim=1)
    rgb = F.normalize(centers + 0.01 * torch.randn(4, 8), dim=1)
    ir = F.normalize(centers + 0.01 * torch.randn(4, 8), dim=1)
    cams = torch.tensor([0, 0, 1, 1])

    rgb_to_ir, ir_to_rgb, metrics = build_gprd_teacher(
        rgb, ir,
        rgb_view2=rgb, ir_view2=ir,
        rgb_cameras=cams, ir_cameras=cams,
        query_per_camera=-1,
        gallery_per_camera=-1,
        positive_count=1,
        hard_negative_count=2,
        mutual_topk=2,
        rerank_k1=2,
        rerank_k2=1,
        rerank_lambda=0.10,
        csls_neighbors=1,
        csls_blend=0.75,
        raw_blend=0.20,
        raw_agreement_topk=2,
        raw_agreement_margin=0.05,
        raw_agreement_min=0.0,
        raw_agreement_rank_weight=0.04,
        teacher_temperature=0.08,
        confidence_floor=0.0,
        entropy_ceiling=1.0,
        stability_rounds=2,
        min_stability=1,
        seed=3)

    assert metrics['stable_edges'] >= 2
    assert metrics['reciprocal_rate'] > 0.0
    assert metrics['mean_raw_agreement'] > 0.0
    assert bool(rgb_to_ir.valid.any().item())
    assert bool(ir_to_rgb.valid.any().item())


def test_gprd_gap_loss_is_zero_without_active_candidates():
    relation = GPRDRelation(
        candidate_ids=torch.full((2, 3), -1, dtype=torch.long),
        teacher_probs=torch.zeros(2, 3),
        positive_mask=torch.zeros(2, 3, dtype=torch.bool),
        negative_mask=torch.zeros(2, 3, dtype=torch.bool),
        raw_agreement=torch.zeros(2, 3),
        confidence=torch.zeros(2),
        valid=torch.zeros(2, dtype=torch.bool),
        positive_count=1,
        hard_negative_count=2,
    )
    student = torch.randn(2, 4, requires_grad=True)
    memory = torch.randn(3, 4)

    loss, stats = gprd_ranking_loss(student, torch.tensor([0, 1]), memory, relation)

    assert loss.item() == 0.0
    assert stats['active_rows'] == 0.0
    assert stats['effective_pairwise'] == 0.0
    assert stats['pull_scale'] == 0.0
    assert stats['row_boost'] == 0.0
    assert stats['neg_teacher_scale'] == 0.0
    assert stats['negative_ceiling'] == 0.0
    assert stats['hard_negative_topk'] == 0.0


def test_gprd_gap_cache_carries_previous_train_edges():
    current = GPRDRelation(
        candidate_ids=torch.tensor([[0, 1, -1], [-1, -1, -1]], dtype=torch.long),
        teacher_probs=torch.tensor([[0.85, 0.15, 0.00], [0.00, 0.00, 0.00]]),
        positive_mask=torch.tensor([[True, False, False], [False, False, False]]),
        negative_mask=torch.tensor([[False, True, False], [False, False, False]]),
        raw_agreement=torch.tensor([[0.90, 0.60, 0.00], [0.00, 0.00, 0.00]]),
        confidence=torch.tensor([0.80, 0.00]),
        valid=torch.tensor([True, False]),
        positive_count=1,
        hard_negative_count=2,
    )
    cached = GPRDRelation(
        candidate_ids=torch.tensor([[2, 3, 0], [2, 3, -1]], dtype=torch.long),
        teacher_probs=torch.tensor([[0.70, 0.25, 0.05], [0.75, 0.25, 0.00]]),
        positive_mask=torch.tensor([[True, False, True], [True, False, False]]),
        negative_mask=torch.tensor([[False, True, False], [False, True, False]]),
        raw_agreement=torch.tensor([[0.95, 0.70, 0.30], [0.90, 0.65, 0.00]]),
        confidence=torch.tensor([0.90, 0.85]),
        valid=torch.tensor([True, True]),
        positive_count=1,
        hard_negative_count=2,
    )

    merged, stats = merge_gprd_teacher_cache(
        current, cached, carry_count=2, carry_weight=0.5, min_confidence=0.2)

    assert merged.candidate_ids.size(1) == 5
    assert merged.valid.tolist() == [True, True]
    assert 2 in merged.candidate_ids[0].tolist()
    assert 3 in merged.candidate_ids[1].tolist()
    assert stats['cache_carried_candidates'] >= 3.0
    assert torch.allclose(
        merged.teacher_probs.sum(dim=1),
        torch.ones(2),
        atol=1e-6)


def test_gprd_gap_loss_filters_low_confidence_rows():
    relation = GPRDRelation(
        candidate_ids=torch.tensor([[0, 1, 2]], dtype=torch.long),
        teacher_probs=torch.tensor([[0.70, 0.20, 0.10]]),
        positive_mask=torch.tensor([[True, False, False]]),
        negative_mask=torch.tensor([[False, True, True]]),
        raw_agreement=torch.tensor([[0.90, 0.60, 0.20]]),
        confidence=torch.tensor([0.20]),
        valid=torch.ones(1, dtype=torch.bool),
        positive_count=1,
        hard_negative_count=2,
    )
    student = torch.randn(1, 4, requires_grad=True)
    memory = torch.randn(3, 4)

    loss, stats = gprd_ranking_loss(
        student, torch.tensor([0]), memory, relation,
        loss_confidence_floor=0.50,
        gap_floor=0.07, gap_weight=0.70,
        positive_floor=0.49, positive_floor_weight=0.12,
        structure_weight=0.06)

    assert loss.item() == 0.0
    assert stats['active_rows'] == 0.0
    assert stats['gap_floor_loss'] == 0.0
    assert stats['structure'] == 0.0


def test_gprd_gap_loss_is_finite_and_backpropagates_with_candidates():
    relation = GPRDRelation(
        candidate_ids=torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.long),
        teacher_probs=torch.tensor([[0.72, 0.18, 0.10], [0.64, 0.24, 0.12]]),
        positive_mask=torch.tensor([[True, True, False], [True, True, False]]),
        negative_mask=torch.tensor([[False, False, True], [False, False, True]]),
        raw_agreement=torch.tensor([[0.90, 0.70, 0.20], [0.80, 0.60, 0.30]]),
        confidence=torch.ones(2),
        valid=torch.ones(2, dtype=torch.bool),
        positive_count=2,
        hard_negative_count=1,
    )
    student = torch.randn(2, 4, requires_grad=True)
    memory = torch.randn(3, 4)

    loss, stats = gprd_ranking_loss(
        student, torch.tensor([0, 1]), memory, relation,
        temperature=0.08, margin=0.04, pairwise_weight=0.12,
        agreement_floor=0.35, conflict_margin=0.02,
        conflict_floor=0.45, negative_teacher_scale=0.35,
        pull_weight=0.16, spread_weight=0.02, adaptive_floor=0.35)
    loss.backward()

    assert torch.isfinite(loss)
    assert stats['active_rows'] == 2.0
    assert stats['raw_agreement'] > 0.0
    assert stats['effective_pairwise'] > 0.0
    assert stats['pull_scale'] >= 1.0
    assert stats['row_boost'] >= 1.0
    assert 0.0 <= stats['neg_teacher_scale'] <= 0.35
    assert stats['negative_ceiling'] >= 0.0
    assert stats['hard_negative_topk'] == 1.0
    assert student.grad is not None
    assert torch.isfinite(student.grad).all()


def test_gprd_gap_loss_reports_explicit_gap_terms():
    relation = GPRDRelation(
        candidate_ids=torch.tensor([[0, 1, 2]], dtype=torch.long),
        teacher_probs=torch.tensor([[0.72, 0.18, 0.10]]),
        positive_mask=torch.tensor([[True, True, False]]),
        negative_mask=torch.tensor([[False, False, True]]),
        raw_agreement=torch.tensor([[0.95, 0.80, 0.30]]),
        confidence=torch.ones(1),
        valid=torch.ones(1, dtype=torch.bool),
        positive_count=2,
        hard_negative_count=1,
    )
    student = torch.tensor([[0.7, 0.7, 0.0, 0.0]], requires_grad=True)
    memory = torch.tensor([
        [0.7, 0.7, 0.0, 0.0],
        [0.6, 0.8, 0.0, 0.0],
        [0.9, 0.5, 0.0, 0.0],
    ])

    loss, stats = gprd_ranking_loss(
        student, torch.tensor([0]), memory, relation,
        temperature=0.08, margin=0.065, pairwise_weight=0.30,
        negative_teacher_scale=0.10, pull_weight=0.34,
        loss_confidence_floor=0.50,
        gap_floor=0.07, gap_weight=0.70,
        positive_floor=0.49, positive_floor_weight=0.12,
        structure_weight=0.06)
    loss.backward()

    assert torch.isfinite(loss)
    assert stats['active_rows'] == 1.0
    assert stats['gap_floor_loss'] > 0.0
    assert stats['positive_floor'] > 0.0
    assert stats['structure'] >= 0.0
    assert student.grad is not None


def test_gprd_gap_loss_filters_conflicts_and_reports_adaptive_scales():
    relation = GPRDRelation(
        candidate_ids=torch.tensor([[0, 1, 2]], dtype=torch.long),
        teacher_probs=torch.tensor([[0.70, 0.20, 0.10]]),
        positive_mask=torch.tensor([[True, False, False]]),
        negative_mask=torch.tensor([[False, True, True]]),
        raw_agreement=torch.tensor([[0.90, 0.60, 0.20]]),
        confidence=torch.ones(1),
        valid=torch.ones(1, dtype=torch.bool),
        positive_count=1,
        hard_negative_count=2,
    )
    student = torch.tensor([[1.0, 0.0, 0.0, 0.0]], requires_grad=True)
    memory = torch.tensor([
        [0.3, 0.7, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.7, 0.7, 0.0, 0.0],
    ])

    loss, stats = gprd_ranking_loss(
        student, torch.tensor([0]), memory, relation,
        temperature=0.08, margin=0.04, pairwise_weight=0.12,
        agreement_floor=0.35, conflict_margin=0.50,
        conflict_floor=0.45, negative_teacher_scale=0.35,
        pull_weight=0.16, spread_weight=0.02, adaptive_floor=0.35)
    loss.backward()

    assert torch.isfinite(loss)
    assert stats['active_rows'] == 1.0
    assert stats['conflict_rate'] > 0.0
    assert stats['reliable_negatives'] >= 1.0
    assert stats['effective_pairwise'] > 0.12
    assert stats['pull_scale'] > 1.0
    assert stats['row_boost'] > 1.0
    assert stats['neg_teacher_scale'] < 0.20
    assert stats['negative_ceiling'] > 0.0
    assert stats['hard_negative_topk'] == 1.0
    assert student.grad is not None
