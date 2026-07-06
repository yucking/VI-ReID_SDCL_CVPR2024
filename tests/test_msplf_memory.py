import importlib.util
import numpy as np
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

_MODULE_PATH = (Path(__file__).resolve().parents[1] /
                'clustercontrast' / 'models' / 'msplf_memory.py')
_SPEC = importlib.util.spec_from_file_location(
    'msplf_memory_under_test', _MODULE_PATH)
_MSPLF_MEMORY = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MSPLF_MEMORY
_SPEC.loader.exec_module(_MSPLF_MEMORY)

_diffuse_cluster_affinity = _MSPLF_MEMORY._diffuse_cluster_affinity
build_msplf_pseudo_labels = _MSPLF_MEMORY.build_msplf_pseudo_labels


def test_msplf_maps_rgb_into_ir_anchor_labels():
    torch.manual_seed(7)
    centers = F.normalize(torch.randn(3, 8), dim=1)
    rgb = torch.cat([
        centers[0].repeat(2, 1) + 0.01 * torch.randn(2, 8),
        centers[1].repeat(2, 1) + 0.01 * torch.randn(2, 8),
        centers[2].repeat(2, 1) + 0.01 * torch.randn(2, 8),
    ], dim=0)
    ir = torch.cat([
        centers[0].repeat(2, 1) + 0.01 * torch.randn(2, 8),
        centers[1].repeat(2, 1) + 0.01 * torch.randn(2, 8),
        centers[2].repeat(2, 1) + 0.01 * torch.randn(2, 8),
    ], dim=0)
    rgb = F.normalize(rgb, dim=1)
    ir = F.normalize(ir, dim=1)

    rgb_labels = np.array([0, 0, 1, 1, 2, 2])
    ir_labels = np.array([10, 10, 11, 11, 12, 12])
    new_rgb, new_ir, metrics = build_msplf_pseudo_labels(
        rgb, ir, rgb_labels, ir_labels,
        rgb_view2=rgb, ir_view2=ir,
        sample_topk=3,
        rgb_cluster_topk=1,
        ir_cluster_topk=1,
        min_pairs=1,
        max_distance=1.0,
        min_margin=-1.0,
        rerank_k1=3,
        rerank_k2=1,
        rerank_lambda=0.10,
        csls_neighbors=2,
        csls_blend=0.75)

    assert metrics['accepted_pairs'] >= 1
    assert metrics['anchor_clusters'] == 3
    assert metrics['unified_clusters'] == 3
    assert np.array_equal(new_ir, ir_labels)
    assert (new_ir >= 0).all()
    assert set(new_rgb[new_rgb >= 0].tolist()).issubset(set(ir_labels.tolist()))
    assert len(set(new_rgb[new_rgb >= 0].tolist()).intersection(set(ir_labels.tolist()))) >= 1
    assert 0.0 <= metrics['neighbor_changed_top1'] <= 1.0
    assert 0.0 <= metrics['propagated_selected'] <= 1.0
    assert metrics['mean_neighbor_score'] >= 0.0


def test_msplf_diffuses_cluster_neighborhood_affinity():
    inf = np.inf
    rgb_to_ir = np.array([
        [0.10, 0.11, inf],
        [0.30, 0.05, inf],
        [0.32, 0.06, inf],
    ], dtype=np.float32)
    ir_to_rgb = np.array([
        [0.10, 0.40, 0.40],
        [0.12, 0.05, 0.06],
        [inf, inf, inf],
    ], dtype=np.float32)

    fused, direct, propagated = _diffuse_cluster_affinity(
        rgb_to_ir, ir_to_rgb,
        rgb_cluster_topk=2,
        ir_cluster_topk=2,
        max_distance=0.50,
        neighborhood_blend=0.50,
        neighborhood_temperature=0.04)

    assert direct[0, 0] > direct[0, 1]
    assert propagated[0, 1] > propagated[0, 0]
    assert fused[0, 1] > direct[0, 1]
    assert np.isclose(fused[0].sum(), 1.0)
