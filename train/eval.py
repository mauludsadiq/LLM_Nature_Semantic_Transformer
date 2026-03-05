"""
Evaluation script — Stage 13.

Usage:
    python train/eval.py \
        --model    train/model.pt \
        --features training_data/features.bin \
        --labels   training_data/labels.bin \
        --splits   training_data/splits.json
"""

import argparse
import json
import struct
import torch
from model import (
    load_model, op_accuracy, tgt_accuracy, per_class_accuracy,
    FEATURE_DIM, OP_NAMES, TGT_NAMES,
)
from train import load_features, load_labels, load_splits


def evaluate(args):
    model    = load_model(args.model)
    features = load_features(args.features)
    op_labels, tgt_labels = load_labels(args.labels)
    splits   = load_splits(args.splits)

    n_train = splits["n_train"]
    n_val   = splits["n_val"]
    n       = len(features)

    sets = {
        "train": (features[:n_train],           op_labels[:n_train],           tgt_labels[:n_train]),
        "val":   (features[n_train:n_train+n_val], op_labels[n_train:n_train+n_val], tgt_labels[n_train:n_train+n_val]),
        "test":  (features[n_train+n_val:],      op_labels[n_train+n_val:],      tgt_labels[n_train+n_val:]),
    }

    print(f"\n{'='*60}")
    print(f"  Evaluation: {args.model}")
    print(f"{'='*60}")

    for split_name, (x, op_lbl, tgt_lbl) in sets.items():
        if len(x) == 0:
            continue
        op_logits, tgt_logits = model(x)
        op_acc  = op_accuracy(op_logits, op_lbl)
        tgt_acc = tgt_accuracy(tgt_logits, tgt_lbl)
        print(f"\n  {split_name:<6} n={len(x):>5}  op_acc={op_acc:.3f}  tgt_acc={tgt_acc:.3f}")

    # Per-class on full set
    op_logits, tgt_logits = model(features)
    per_cls = per_class_accuracy(op_logits, op_labels)
    print(f"\n  Per-class accuracy (full set):")
    for name, acc in per_cls.items():
        if acc is None:
            print(f"    {name:<20}  n/a")
        else:
            bar = "█" * int(acc * 20)
            print(f"    {name:<20}  {acc:.3f}  {bar}")

    overall = op_accuracy(op_logits, op_labels)
    gate = "✓ PASSED" if overall >= 0.90 else "✗ FAILED"
    print(f"\n  Overall op_acc={overall:.3f}  Stage 13 gate: {gate}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    default="train/model.pt")
    parser.add_argument("--features", default="training_data/features.bin")
    parser.add_argument("--labels",   default="training_data/labels.bin")
    parser.add_argument("--splits",   default="training_data/splits.json")
    args = parser.parse_args()
    evaluate(args)
