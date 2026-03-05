"""
Evaluation script — Stage 15.
Evaluates the ONNX model via onnxruntime.

Usage:
    python train/eval.py \
        --model    train/model.onnx \
        --features training_data/features.bin \
        --labels   training_data/labels.bin \
        --splits   training_data/splits.json
"""

import argparse, json, struct
import numpy as np
import onnxruntime as ort

FDIM = 25
OP_NAMES  = ["SELECT_UNIVERSE","WITNESS_NEAREST","ATTEND","FFN_STEP",
             "PROJECT_LAYER","RETURN_SET","ACCEPT","REJECT"]
TGT_NAMES = ["PHONEME","SYLLABLE","MORPHEME","WORD","PHRASE","SEMANTIC","DISCOURSE","none"]


def load_features(path):
    data = open(path,"rb").read()
    n = len(data) // (FDIM * 4)
    return np.frombuffer(data, dtype=np.float32).reshape(n, FDIM)


def load_labels(path):
    data = open(path,"rb").read()
    n = len(data) // 2
    op  = np.array([data[i*2]   for i in range(n)], dtype=np.int64)
    tgt = np.array([data[i*2+1] for i in range(n)], dtype=np.int64)
    return op, tgt


def evaluate(args):
    print(f"Loading ONNX model: {args.model}")
    sess = ort.InferenceSession(args.model)

    features           = load_features(args.features)
    op_labels, tgt_labels = load_labels(args.labels)
    splits             = json.load(open(args.splits))

    n       = len(features)
    n_train = splits["n_train"]
    n_val   = splits["n_val"]

    sets = {
        "train": (features[:n_train],              op_labels[:n_train],              tgt_labels[:n_train]),
        "val":   (features[n_train:n_train+n_val], op_labels[n_train:n_train+n_val], tgt_labels[n_train:n_train+n_val]),
        "test":  (features[n_train+n_val:],        op_labels[n_train+n_val:],        tgt_labels[n_train+n_val:]),
    }

    print(f"\n{'='*62}")
    print(f"  Evaluation: {args.model}")
    print(f"{'='*62}")

    for split_name, (x, op_lbl, tgt_lbl) in sets.items():
        if len(x) == 0: continue
        op_logits, tgt_logits = sess.run(None, {"features": x.astype(np.float32)})
        op_pred  = op_logits.argmax(axis=1)
        tgt_pred = tgt_logits.argmax(axis=1)
        op_acc   = (op_pred == op_lbl).mean()
        mask     = tgt_lbl != 7
        tgt_acc  = (tgt_pred[mask] == tgt_lbl[mask]).mean() if mask.any() else 1.0
        print(f"  {split_name:<6} n={len(x):>5}  op_acc={op_acc:.3f}  tgt_acc={tgt_acc:.3f}")

    # Full set per-class
    op_logits, _ = sess.run(None, {"features": features.astype(np.float32)})
    op_pred = op_logits.argmax(axis=1)
    overall = (op_pred == op_labels).mean()

    print(f"\n  Per-class accuracy (full set, n={n}):")
    for i, name in enumerate(OP_NAMES):
        mask = op_labels == i
        if not mask.any(): continue
        acc = (op_pred[mask] == i).mean()
        bar = "█" * int(acc * 20)
        print(f"    [{i}] {name:<20}  n={mask.sum():>5}  {acc:.3f}  {bar}")

    gate = "✓ PASSED" if overall >= 0.90 else "✗ FAILED"
    print(f"\n  Overall op_acc={overall:.3f}  Stage 15 gate (>=0.90): {gate}")
    print(f"  Acceptance rate (op_acc as proxy): {overall*100:.1f}%")
    stage15 = "✓ PASSED" if overall >= 0.85 else "✗ FAILED"
    print(f"  Stage 15 gate  (acceptance >=85%): {stage15}")
    print(f"{'='*62}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    default="train/model.onnx")
    parser.add_argument("--features", default="training_data/features.bin")
    parser.add_argument("--labels",   default="training_data/labels.bin")
    parser.add_argument("--splits",   default="training_data/splits.json")
    evaluate(parser.parse_args())
