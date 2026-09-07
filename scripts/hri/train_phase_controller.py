#!/usr/bin/env python
"""Train a small phase classifier from HRI action traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.hri.phase_controller import (  # noqa: E402
    FEATURE_NAMES,
    find_trace_files,
    load_trace_examples,
    predict_batch,
    save_model,
    train_nearest_centroid,
)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    labels = sorted(set(map(str, y_true)) | set(map(str, y_pred)))
    correct = int(np.sum(y_true == y_pred))
    total = int(len(y_true))
    per_label = {}
    for label in labels:
        mask = y_true == label
        denom = int(np.sum(mask))
        if denom:
            per_label[label] = {
                "support": denom,
                "accuracy": float(np.mean(y_pred[mask] == label)),
            }
    confusion = {label: {other: 0 for other in labels} for label in labels}
    for t, p in zip(y_true, y_pred):
        confusion[str(t)][str(p)] += 1
    return {
        "accuracy": float(correct / total) if total else 0.0,
        "correct": correct,
        "total": total,
        "per_label": per_label,
        "confusion": confusion,
    }


def _class_counts(y: np.ndarray) -> dict[str, int]:
    out = {}
    for label in sorted(set(map(str, y))):
        out[label] = int(np.sum(y == label))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trace-root",
        action="append",
        required=True,
        help="Trace file or directory. May be passed multiple times.",
    )
    parser.add_argument("--output", required=True, help="Pickle checkpoint path.")
    parser.add_argument(
        "--metrics-json",
        default=None,
        help="Metrics JSON path. Default: <output>.metrics.json",
    )
    parser.add_argument(
        "--model",
        choices=["nearest_centroid"],
        default="nearest_centroid",
        help="Classifier family. Pure NumPy nearest-centroid is dependency-free.",
    )
    parser.add_argument("--heldout-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument(
        "--drop-visual",
        action="store_true",
        help="Zero visual-progress features during training. Use this when "
             "runtime eval will not load a pace visual model.",
    )
    args = parser.parse_args()

    trace_files = find_trace_files(args.trace_root)
    if args.max_files is not None:
        trace_files = trace_files[: max(0, int(args.max_files))]
    if not trace_files:
        raise SystemExit("No *_trace.json files found")

    rng = random.Random(args.seed)
    shuffled = list(trace_files)
    rng.shuffle(shuffled)
    n_heldout = int(round(len(shuffled) * float(args.heldout_frac)))
    if len(shuffled) > 1:
        n_heldout = min(max(1, n_heldout), len(shuffled) - 1)
    else:
        n_heldout = 0
    heldout_files = shuffled[:n_heldout]
    train_files = shuffled[n_heldout:] or shuffled

    x_train, y_train, used_train = load_trace_examples(
        train_files,
        drop_visual=args.drop_visual,
    )
    x_test, y_test, used_test = load_trace_examples(
        heldout_files,
        drop_visual=args.drop_visual,
    )
    if len(y_train) == 0:
        raise SystemExit("No labeled rows found in training traces")

    model = train_nearest_centroid(x_train, y_train)
    model["train_trace_files"] = used_train
    model["heldout_trace_files"] = used_test
    model["feature_names"] = list(FEATURE_NAMES)
    model["training"] = {
        "trace_roots": args.trace_root,
        "n_trace_files": len(trace_files),
        "n_train_files": len(used_train),
        "n_heldout_files": len(used_test),
        "n_train_rows": int(len(y_train)),
        "n_heldout_rows": int(len(y_test)),
        "train_class_counts": _class_counts(y_train),
        "heldout_class_counts": _class_counts(y_test),
        "drop_visual": bool(args.drop_visual),
    }

    train_pred = predict_batch(model, x_train)
    metrics = {
        "model_type": model["model_type"],
        "feature_names": list(FEATURE_NAMES),
        "train": _metrics(y_train, train_pred),
        "training": model["training"],
    }
    if len(y_test):
        test_pred = predict_batch(model, x_test)
        metrics["heldout"] = _metrics(y_test, test_pred)
    else:
        metrics["heldout"] = None

    save_model(args.output, model)
    metrics_path = Path(args.metrics_json) if args.metrics_json else Path(args.output).with_suffix(".metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True))

    heldout_acc = metrics["heldout"]["accuracy"] if metrics["heldout"] else None
    print(f"saved controller: {args.output}")
    print(f"saved metrics:    {metrics_path}")
    print(f"train rows: {len(y_train)}  files: {len(used_train)}  accuracy: {metrics['train']['accuracy']:.3f}")
    if heldout_acc is not None:
        print(f"heldout rows: {len(y_test)}  files: {len(used_test)}  accuracy: {heldout_acc:.3f}")
    print(f"train class counts: {model['training']['train_class_counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
