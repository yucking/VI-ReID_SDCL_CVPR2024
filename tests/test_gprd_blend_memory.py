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
                'clustercontrast' / 'models' / 'gprd_blend_memory.py')
_SPEC = importlib.util.spec_from_file_location(
    'gprd_blend_memory_under_test', _MODULE_PATH)
_gprd_blend_memory = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _gprd_blend_memory
_SPEC.loader.exec_module(_gprd_blend_memory)

GPRDRelation = _gprd_blend_memory.GPRDRelation
build_gprd_teacher = _gprd_blend_memory.build_gprd_teacher
gprd_ranking_loss = _gprd_blend_memory.gprd_ranking_loss


def test_gprd_teacher_keeps_stable_reciprocal_edges():
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
        raw_blend=0.35,
        teacher_temperature=0.05,
        confidence_floor=0.0,
        entropy_ceiling=1.0,
        stability_rounds=2,
        min_stability=2,
        seed=3)

    assert metrics['stable_edges'] >= 2
    assert metrics['reciprocal_rate'] > 0.0
    assert metrics['camera_balance_entropy'] > 0.0
    assert metrics['raw_blend'] == 0.35
    assert bool(rgb_to_ir.valid.any().item())
    assert bool(ir_to_rgb.valid.any().item())


def test_gprd_loss_is_zero_without_active_candidates():
    relation = GPRDRelation(
        candidate_ids=torch.full((2, 3), -1, dtype=torch.long),
        teacher_probs=torch.zeros(2, 3),
        positive_mask=torch.zeros(2, 3, dtype=torch.bool),
        negative_mask=torch.zeros(2, 3, dtype=torch.bool),
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


def test_gprd_loss_is_finite_and_backpropagates_with_candidates():
    relation = GPRDRelation(
        candidate_ids=torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.long),
        teacher_probs=torch.tensor([[0.8, 0.2, 0.0], [0.7, 0.3, 0.0]]),
        positive_mask=torch.tensor([[True, True, False], [True, True, False]]),
        negative_mask=torch.tensor([[False, False, True], [False, False, True]]),
        confidence=torch.ones(2),
        valid=torch.ones(2, dtype=torch.bool),
        positive_count=2,
        hard_negative_count=1,
    )
    student = torch.randn(2, 4, requires_grad=True)
    memory = torch.randn(3, 4)

    loss, stats = gprd_ranking_loss(
        student, torch.tensor([0, 1]), memory, relation,
        temperature=0.08, margin=0.04, pairwise_weight=0.35)
    loss.backward()

    assert torch.isfinite(loss)
    assert stats['active_rows'] == 2.0
    assert 'pull' in stats and 'margin' in stats
    assert 'pos_sim' in stats and 'neg_sim' in stats
    assert 'raw_support' in stats and 'gap' in stats
    assert student.grad is not None
    assert torch.isfinite(student.grad).all()


if __name__ == '__main__':
    test_gprd_teacher_keeps_stable_reciprocal_edges()
    test_gprd_loss_is_zero_without_active_candidates()
    test_gprd_loss_is_finite_and_backpropagates_with_candidates()
    print('GPRD-BLEND smoke test passed')
