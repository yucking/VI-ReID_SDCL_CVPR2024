from __future__ import print_function, absolute_import

"""Test-time metric stack for a frozen SYSU checkpoint.

The script deliberately stays outside the training path.  It caches the two
existing descriptor branches, verifies the raw protocol result, then evaluates
predefined feature fusion and retrieval post-processing candidates.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import os.path as osp
import time

import numpy as np
import torch
from torch import nn
import torch.utils.data as data

from config import cfg
from clustercontrast.model_vit_cmrefine import make_model
from clustercontrast.utils.data import transforms as T
from clustercontrast.utils.serialization import load_checkpoint
from clustercontrast.utils.rerank import re_ranking
from test_sysu import (
    TestData,
    eval_sysu,
    fliplr,
    process_gallery_sysu,
    process_query_sysu,
)


EXPECTED_REPRODUCTION = {
    "all": {"mAP": 0.6341, "mINP": 0.5038},
    "indoor": {"mAP": 0.7658, "mINP": 0.7312},
}
TARGETS = {
    "all": {"mAP": 0.6324, "mINP": 0.5106},
    "indoor": {"mAP": 0.7690, "mINP": 0.7350},
}


def parse_float_list(value):
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one numeric value")
    return [float(item) for item in values]


def parse_int_list(value):
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer value")
    return [int(item) for item in values]


def checkpoint_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize(features):
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.clip(norms, 1e-12, None)


def build_test_transform(height, width):
    normalizer = T.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    return T.Compose([
        T.ToPILImage(),
        T.Resize((height, width)),
        T.ToTensor(),
        normalizer,
    ])


def build_loader(images, labels, transform, width, height, batch_size, workers):
    dataset = TestData(images, labels, transform=transform, img_size=(width, height))
    return data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )


def extract_dual_features(model, loader, modal, name):
    """Return independently normalized global and local descriptors."""
    model.eval()
    global_features = []
    local_features = []
    started = time.time()

    with torch.no_grad():
        for batch_index, (images, _) in enumerate(loader):
            flipped = fliplr(images)
            images = images.cuda(non_blocking=True)
            flipped = flipped.cuda(non_blocking=True)

            global_feat, local_feat = model(images, images, modal)
            global_flip, local_flip = model(flipped, flipped, modal)

            global_feat = torch.nn.functional.normalize(
                (global_feat + global_flip) / 2.0, dim=1
            )
            local_feat = torch.nn.functional.normalize(
                (local_feat + local_flip) / 2.0, dim=1
            )
            global_features.append(global_feat.cpu().numpy().astype(np.float32))
            local_features.append(local_feat.cpu().numpy().astype(np.float32))

            if (batch_index + 1) % 20 == 0:
                print("[EXTRACT] {} batch={}/{}".format(
                    name, batch_index + 1, len(loader)
                ))

    global_features = np.concatenate(global_features, axis=0)
    local_features = np.concatenate(local_features, axis=0)
    print("[EXTRACT] {} complete: n={} elapsed={:.1f}s".format(
        name, len(global_features), time.time() - started
    ))
    return global_features, local_features


def cache_features(args, cache_path, checkpoint_hash):
    transform = build_test_transform(args.height, args.width)
    model = make_model(cfg, num_class=0, camera_num=0, view_num=0).cuda()
    model = nn.DataParallel(model)
    checkpoint = load_checkpoint(args.checkpoint)
    model.load_state_dict(checkpoint["state_dict"])

    payload = {}
    query_images, query_labels, query_cams = process_query_sysu(args.data_dir, mode="all")
    query_loader = build_loader(
        query_images, query_labels, transform, args.width, args.height,
        args.batch_size, args.workers,
    )
    query_global, query_local = extract_dual_features(
        model, query_loader, modal=2, name="query-ir"
    )
    payload["query_global"] = query_global
    payload["query_local"] = query_local
    payload["query_labels"] = query_labels.astype(np.int32)
    payload["query_cams"] = query_cams.astype(np.int32)

    for mode in ("all", "indoor"):
        for trial in range(10):
            gallery_images, gallery_labels, gallery_cams = process_gallery_sysu(
                args.data_dir, mode=mode, trial=trial
            )
            gallery_loader = build_loader(
                gallery_images, gallery_labels, transform, args.width, args.height,
                args.batch_size, args.workers,
            )
            gallery_global, gallery_local = extract_dual_features(
                model, gallery_loader, modal=1,
                name="{}-gallery-trial{}".format(mode, trial),
            )
            prefix = "{}_{}".format(mode, trial)
            payload[prefix + "_global"] = gallery_global
            payload[prefix + "_local"] = gallery_local
            payload[prefix + "_labels"] = gallery_labels.astype(np.int32)
            payload[prefix + "_cams"] = gallery_cams.astype(np.int32)

    np.savez(cache_path, **payload)
    metadata = {
        "checkpoint": osp.abspath(args.checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "feature_format": "global_local_l2_normalized_v1",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(cache_path + ".json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=True)
    print("[CACHE] wrote {}".format(cache_path))
    return payload


def load_or_extract_features(args):
    checkpoint_hash = checkpoint_sha256(args.checkpoint)
    os.makedirs(args.cache_dir, exist_ok=True)
    cache_path = osp.join(args.cache_dir, "sysu_{}.npz".format(checkpoint_hash[:16]))

    if osp.isfile(cache_path) and not args.force_extract:
        print("[CACHE] loading {}".format(cache_path))
        with np.load(cache_path, allow_pickle=False) as cached:
            return {key: cached[key] for key in cached.files}, checkpoint_hash, cache_path

    return cache_features(args, cache_path, checkpoint_hash), checkpoint_hash, cache_path


def fuse_features(global_features, local_features, global_weight):
    local_weight = 1.0 - global_weight
    if not 0.0 < global_weight < 1.0:
        raise ValueError("global feature weight must be in (0, 1)")
    return np.concatenate([
        math.sqrt(global_weight) * normalize(global_features),
        math.sqrt(local_weight) * normalize(local_features),
    ], axis=1)


def cosine_distance(query_features, gallery_features):
    return 1.0 - np.matmul(query_features, gallery_features.T)


def row_minmax(distance):
    minimum = distance.min(axis=1, keepdims=True)
    maximum = distance.max(axis=1, keepdims=True)
    return (distance - minimum) / np.clip(maximum - minimum, 1e-12, None)


def csls_distance(query_features, gallery_features, neighbors):
    """Cross-domain local scaling to suppress gallery hubs without labels."""
    similarity = np.matmul(query_features, gallery_features.T)
    query_k = min(neighbors, similarity.shape[1])
    gallery_k = min(neighbors, similarity.shape[0])
    query_scale = np.partition(
        similarity, similarity.shape[1] - query_k, axis=1
    )[:, -query_k:].mean(axis=1)
    gallery_scale = np.partition(
        similarity.T, similarity.shape[0] - gallery_k, axis=1
    )[:, -gallery_k:].mean(axis=1)
    return -(2.0 * similarity - query_scale[:, None] - gallery_scale[None, :])


def average_mode(payload, mode, config):
    query_features = fuse_features(
        payload["query_global"], payload["query_local"], config["global_weight"]
    )
    query_labels = payload["query_labels"]
    query_cams = payload["query_cams"]
    cmcs = []
    maps = []
    minps = []

    for trial in range(10):
        prefix = "{}_{}".format(mode, trial)
        gallery_features = fuse_features(
            payload[prefix + "_global"], payload[prefix + "_local"],
            config["global_weight"],
        )
        gallery_labels = payload[prefix + "_labels"]
        gallery_cams = payload[prefix + "_cams"]
        base_distance = cosine_distance(query_features, gallery_features)
        distance = base_distance
        csls = None
        if config["csls_neighbors"] is not None:
            csls = csls_distance(query_features, gallery_features, config["csls_neighbors"])

        if config["rerank"]:
            query_distance = cosine_distance(query_features, query_features)
            gallery_distance = cosine_distance(gallery_features, gallery_features)
            distance = re_ranking(
                base_distance,
                query_distance,
                gallery_distance,
                k1=config["rerank_k1"],
                k2=config["rerank_k2"],
                lambda_value=config["rerank_lambda"],
            )
            if csls is not None:
                distance = (
                    config["csls_blend"] * row_minmax(distance)
                    + (1.0 - config["csls_blend"]) * row_minmax(csls)
                )
        elif csls is not None:
            distance = csls

        cmc, mAP, mINP = eval_sysu(
            distance, query_labels, gallery_labels, query_cams, gallery_cams
        )
        cmcs.append(cmc)
        maps.append(mAP)
        minps.append(mINP)

    return {
        "rank1": float(np.mean([cmc[0] for cmc in cmcs])),
        "mAP": float(np.mean(maps)),
        "mINP": float(np.mean(minps)),
    }


def target_pass(metrics):
    return all(
        metrics[mode][key] >= TARGETS[mode][key]
        for mode in TARGETS
        for key in TARGETS[mode]
    )


def target_gap(metrics):
    return sum(
        max(0.0, TARGETS[mode][key] - metrics[mode][key])
        for mode in TARGETS
        for key in TARGETS[mode]
    )


def composite_score(metrics):
    return (
        0.20 * metrics["all"]["mAP"]
        + 0.35 * metrics["all"]["mINP"]
        + 0.20 * metrics["indoor"]["mAP"]
        + 0.25 * metrics["indoor"]["mINP"]
    )


def evaluate_candidate(payload, config, phase):
    metrics = {mode: average_mode(payload, mode, config) for mode in ("all", "indoor")}
    result = dict(config)
    result.update({
        "phase": phase,
        "all_rank1": metrics["all"]["rank1"],
        "all_mAP": metrics["all"]["mAP"],
        "all_mINP": metrics["all"]["mINP"],
        "indoor_rank1": metrics["indoor"]["rank1"],
        "indoor_mAP": metrics["indoor"]["mAP"],
        "indoor_mINP": metrics["indoor"]["mINP"],
        "target_pass": target_pass(metrics),
        "target_gap": target_gap(metrics),
        "composite": composite_score(metrics),
    })
    print(
        "[METRIC] phase={phase} w={global_weight:.2f} rr={rerank} "
        "k1={rerank_k1} k2={rerank_k2} lambda={rerank_lambda:.2f} "
        "csls={csls_neighbors} blend={csls_blend:.2f} "
        "all={all_mAP:.2%}/{all_mINP:.2%} indoor={indoor_mAP:.2%}/{indoor_mINP:.2%} "
        "target={target}".format(
            target="PASS" if result["target_pass"] else "MISS", **result
        )
    )
    return result


def rank_results(results):
    return sorted(
        results,
        key=lambda item: (
            not item["target_pass"],
            item["target_gap"],
            -item["composite"],
        ),
    )


def write_results(results, output_dir):
    path = osp.join(output_dir, "metric_results.csv")
    fieldnames = list(results[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rank_results(results))
    print("[RESULT] wrote {}".format(path))


def baseline_config():
    return {
        "global_weight": 0.5,
        "rerank": False,
        "rerank_k1": 0,
        "rerank_k2": 0,
        "rerank_lambda": 0.0,
        "csls_neighbors": None,
        "csls_blend": 1.0,
    }


def fixed_config(args):
    return {
        "global_weight": args.fixed_global_weight,
        "rerank": True,
        "rerank_k1": args.fixed_rerank_k1,
        "rerank_k2": args.fixed_rerank_k2,
        "rerank_lambda": args.fixed_rerank_lambda,
        "csls_neighbors": args.fixed_csls_neighbors,
        "csls_blend": args.fixed_csls_blend,
    }


def assert_baseline(baseline, tolerance):
    mismatches = []
    for mode, expected in EXPECTED_REPRODUCTION.items():
        for key, value in expected.items():
            actual = baseline["{}_{}".format(mode, key)]
            if abs(actual - value) > tolerance:
                mismatches.append(
                    "{} {} expected {:.2%}, got {:.2%}".format(mode, key, value, actual)
                )
    if mismatches:
        raise RuntimeError(
            "Raw checkpoint result does not match the saved reproduction log. "
            "Do not interpret post-processing results. " + "; ".join(mismatches)
        )
    print("[BASELINE] reproduction check passed within {:.2%}".format(tolerance))


def print_best_result(results):
    best = rank_results(results)[0]
    print(
        "[METRIC-BEST] phase={phase} target={target} all={all_mAP:.2%}/{all_mINP:.2%} "
        "indoor={indoor_mAP:.2%}/{indoor_mINP:.2%} config={config}".format(
            target="PASS" if best["target_pass"] else "MISS",
            config=json.dumps({
                key: best[key] for key in (
                    "global_weight", "rerank", "rerank_k1", "rerank_k2",
                    "rerank_lambda", "csls_neighbors", "csls_blend",
                )
            }, sort_keys=True),
            **best
        )
    )


def main():
    parser = argparse.ArgumentParser(description="SYSU frozen-checkpoint metric stack")
    parser.add_argument("--config-file", default="vit_base_ics_288.yml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--height", type=int, default=288)
    parser.add_argument("--width", type=int, default=144)
    parser.add_argument("--force-extract", action="store_true")
    parser.add_argument("--baseline-tolerance", type=float, default=0.005)
    parser.add_argument("--fixed", action="store_true",
                        help="evaluate one frozen metric-stack configuration without a sweep")
    parser.add_argument("--skip-baseline-check", action="store_true",
                        help="allow a different checkpoint while still reporting its raw baseline")
    parser.add_argument("--fixed-global-weight", type=float, default=0.25)
    parser.add_argument("--fixed-rerank-k1", type=int, default=30)
    parser.add_argument("--fixed-rerank-k2", type=int, default=3)
    parser.add_argument("--fixed-rerank-lambda", type=float, default=0.10)
    parser.add_argument("--fixed-csls-neighbors", type=int, default=5)
    parser.add_argument("--fixed-csls-blend", type=float, default=0.75)
    parser.add_argument("--global-weights", type=parse_float_list, default=parse_float_list("0.25,0.40,0.50,0.60,0.75"))
    parser.add_argument("--rerank-k1", type=parse_int_list, default=parse_int_list("15,20,30"))
    parser.add_argument("--rerank-k2", type=parse_int_list, default=parse_int_list("1,3"))
    parser.add_argument("--rerank-lambda", type=parse_float_list, default=parse_float_list("0.10,0.20,0.30"))
    parser.add_argument("--csls-neighbors", type=parse_int_list, default=parse_int_list("5,10"))
    parser.add_argument("--csls-blends", type=parse_float_list, default=parse_float_list("0.50,0.75"))
    args = parser.parse_args()

    if not osp.isfile(args.checkpoint):
        raise FileNotFoundError("checkpoint not found: {}".format(args.checkpoint))
    if not osp.isdir(args.data_dir):
        raise FileNotFoundError("SYSU data directory not found: {}".format(args.data_dir))
    if args.cache_dir is None:
        args.cache_dir = osp.join(args.output_dir, "feature_cache")
    os.makedirs(args.output_dir, exist_ok=True)

    cfg.merge_from_file(args.config_file)
    cfg.freeze()
    torch.backends.cudnn.benchmark = True

    payload, checkpoint_hash, cache_path = load_or_extract_features(args)
    print("[CHECKPOINT] {} sha256={}".format(osp.abspath(args.checkpoint), checkpoint_hash))
    print("[CACHE] {}".format(cache_path))

    results = []
    baseline = evaluate_candidate(payload, baseline_config(), phase="baseline")
    results.append(baseline)
    if not args.skip_baseline_check:
        assert_baseline(baseline, args.baseline_tolerance)

    if args.fixed:
        results.append(evaluate_candidate(payload, fixed_config(args), phase="fixed"))
        write_results(results, args.output_dir)
        print_best_result(results)
        return

    stage_one = []
    for weight in args.global_weights:
        config = baseline_config()
        config["global_weight"] = weight
        stage_one.append(evaluate_candidate(payload, config, phase="fusion"))
        for neighbors in args.csls_neighbors:
            config = baseline_config()
            config["global_weight"] = weight
            config["csls_neighbors"] = neighbors
            stage_one.append(evaluate_candidate(payload, config, phase="fusion_csls"))
    results.extend(stage_one)

    selected_weights = []
    for result in rank_results(stage_one):
        weight = result["global_weight"]
        if weight not in selected_weights:
            selected_weights.append(weight)
        if len(selected_weights) == 2:
            break
    print("[SEARCH] selected global weights for re-ranking: {}".format(selected_weights))

    stage_two = []
    for weight in selected_weights:
        for k1 in args.rerank_k1:
            for k2 in args.rerank_k2:
                for lambda_value in args.rerank_lambda:
                    config = baseline_config()
                    config.update({
                        "global_weight": weight,
                        "rerank": True,
                        "rerank_k1": k1,
                        "rerank_k2": k2,
                        "rerank_lambda": lambda_value,
                    })
                    stage_two.append(evaluate_candidate(payload, config, phase="rerank"))
    results.extend(stage_two)

    stage_three = []
    for seed_result in rank_results(stage_two)[:4]:
        for neighbors in args.csls_neighbors:
            for blend in args.csls_blends:
                config = baseline_config()
                config.update({
                    "global_weight": seed_result["global_weight"],
                    "rerank": True,
                    "rerank_k1": seed_result["rerank_k1"],
                    "rerank_k2": seed_result["rerank_k2"],
                    "rerank_lambda": seed_result["rerank_lambda"],
                    "csls_neighbors": neighbors,
                    "csls_blend": blend,
                })
                stage_three.append(evaluate_candidate(payload, config, phase="rerank_csls"))
    results.extend(stage_three)

    write_results(results, args.output_dir)
    print_best_result(results)


if __name__ == "__main__":
    main()
