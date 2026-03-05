"""
Training script — Stage 13.

Usage:
    python train/train.py \
        --features training_data/features.bin \
        --labels   training_data/labels.bin \
        --splits   training_data/splits.json \
        --out      train/model.pt \
        --epochs   50 \
        --batch    64 \
        --lr       1e-3 \
        --seed     42

Gate: val op_accuracy >= 0.90
"""

import argparse
import json
import math
import os
import struct
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Subset

from model import (
    ProposerModel, ProposerLoss, save_model,
    op_accuracy, tgt_accuracy, per_class_accuracy,
    FEATURE_DIM, OP_NAMES,
)


# ── Data loading ───────────────────────────────────────────────────────────────

def load_features(path: str) -> torch.Tensor:
    data = open(path, "rb").read()
    n    = len(data) // (FEATURE_DIM * 4)
    flat = struct.unpack(f"<{n * FEATURE_DIM}f", data)
    return torch.tensor(flat, dtype=torch.float32).view(n, FEATURE_DIM)


def load_labels(path: str) -> tuple:
    data = open(path, "rb").read()
    n    = len(data) // 2
    op_labels  = torch.tensor([data[i*2]   for i in range(n)], dtype=torch.long)
    tgt_labels = torch.tensor([data[i*2+1] for i in range(n)], dtype=torch.long)
    return op_labels, tgt_labels


def load_splits(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


# ── Training loop ──────────────────────────────────────────────────────────────

def train(args):
    torch.manual_seed(args.seed)

    print(f"Loading data from {args.features} / {args.labels}")
    features   = load_features(args.features)
    op_labels, tgt_labels = load_labels(args.labels)
    splits     = load_splits(args.splits)

    n = len(features)
    print(f"  {n} records, feature_dim={FEATURE_DIM}")

    # Build dataset and splits
    dataset = TensorDataset(features, op_labels, tgt_labels)

    # Use stored split indices (deterministic)
    train_idx = splits["n_train"]
    val_idx   = splits["n_val"]
    # Simple sequential split (indices are pre-shuffled in splits.json)
    train_set = Subset(dataset, list(range(train_idx)))
    val_set   = Subset(dataset, list(range(train_idx, train_idx + val_idx)))
    test_set  = Subset(dataset, list(range(train_idx + val_idx, n)))

    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=args.batch, shuffle=False)

    print(f"  train={len(train_set)}  val={len(val_set)}  test={len(test_set)}")

    # Class weights from splits.json
    class_weights = torch.tensor(splits["class_weights"], dtype=torch.float32)
    print(f"  class_weights: {[f'{w:.3f}' for w in class_weights.tolist()]}")

    # Model
    model   = ProposerModel()
    loss_fn = ProposerLoss(class_weights=class_weights)
    optim   = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched   = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode='min', factor=0.5, patience=3
    )

    print(f"\nModel: {model.n_params()} parameters")
    print(f"Training for {args.epochs} epochs, batch={args.batch}, lr={args.lr}\n")

    best_val_acc  = 0.0
    best_epoch    = 0
    patience_left = args.patience

    for epoch in range(1, args.epochs + 1):
        # ── Train ──────────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        train_acc  = 0.0
        n_batches  = 0

        for x, op_lbl, tgt_lbl in train_loader:
            optim.zero_grad()
            op_logits, tgt_logits = model(x)
            loss, _, _ = loss_fn(op_logits, tgt_logits, op_lbl, tgt_lbl)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            train_loss += loss.item()
            train_acc  += op_accuracy(op_logits, op_lbl)
            n_batches  += 1

        train_loss /= n_batches
        train_acc  /= n_batches

        # ── Validate ───────────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        val_acc  = 0.0
        n_val    = 0

        with torch.no_grad():
            for x, op_lbl, tgt_lbl in val_loader:
                op_logits, tgt_logits = model(x)
                loss, _, _ = loss_fn(op_logits, tgt_logits, op_lbl, tgt_lbl)
                val_loss += loss.item()
                val_acc  += op_accuracy(op_logits, op_lbl)
                n_val    += 1

        val_loss /= max(n_val, 1)
        val_acc  /= max(n_val, 1)

        sched.step(val_loss)

        gate = "✓" if val_acc >= 0.90 else " "
        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"train_loss={train_loss:.4f}  train_acc={train_acc:.3f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f} {gate}")

        # ── Early stopping ────────────────────────────────────────────────────
        if val_acc > best_val_acc:
            best_val_acc  = val_acc
            best_epoch    = epoch
            patience_left = args.patience
            save_model(model, args.out)
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"\nEarly stopping at epoch {epoch} (patience={args.patience})")
                break

    print(f"\nBest val_acc={best_val_acc:.3f} at epoch {best_epoch}")

    # ── Stage 13 gate ─────────────────────────────────────────────────────────
    if best_val_acc >= 0.90:
        print(f"Stage 13 gate: ✓ PASSED (val_acc={best_val_acc:.3f} >= 0.90)")
    else:
        print(f"Stage 13 gate: ✗ FAILED (val_acc={best_val_acc:.3f} < 0.90)")
        print("  Consider: more epochs, lower lr, larger hidden_dim, more corpus data")

    return best_val_acc


# ── Per-class report ───────────────────────────────────────────────────────────

def report(model_path: str, features_path: str, labels_path: str):
    from model import load_model
    model    = load_model(model_path)
    features = load_features(features_path)
    op_labels, tgt_labels = load_labels(labels_path)

    op_logits, tgt_logits = model(features)
    overall = op_accuracy(op_logits, op_labels)
    per_cls = per_class_accuracy(op_logits, op_labels)

    print(f"\nPer-class accuracy (overall={overall:.3f}):")
    for name, acc in per_cls.items():
        if acc is None:
            print(f"  {name:<20}  n/a")
        else:
            bar = "█" * int(acc * 20)
            print(f"  {name:<20}  {acc:.3f}  {bar}")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the learned proposer")
    parser.add_argument("--features", default="training_data/features.bin")
    parser.add_argument("--labels",   default="training_data/labels.bin")
    parser.add_argument("--splits",   default="training_data/splits.json")
    parser.add_argument("--out",      default="train/model.pt")
    parser.add_argument("--epochs",   type=int,   default=50)
    parser.add_argument("--batch",    type=int,   default=64)
    parser.add_argument("--lr",       type=float, default=1e-3)
    parser.add_argument("--patience", type=int,   default=5)
    parser.add_argument("--seed",     type=int,   default=42)
    parser.add_argument("--report",   action="store_true")
    args = parser.parse_args()

    if args.report:
        report(args.out, args.features, args.labels)
    else:
        train(args)
