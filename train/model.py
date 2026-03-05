"""
Learned proposer model — Stage 13 of the Phase 2 blueprint.

Architecture:
    Linear(32 → 64) + ReLU + Dropout(0.1)
    Linear(64 → 64) + ReLU + Dropout(0.1)
    Head A: Linear(64 → 8)   # op_kind logits
    Head B: Linear(64 → 8)   # tgt_layer logits

Loss:
    L = CrossEntropy(op_kind, weight=class_weights) + 0.5 * CrossEntropy(tgt_layer)

Input:  [N, 32] float32 feature vectors (from FeatureEncoder)
Output: op_logits [N, 8], tgt_logits [N, 8]

Op classes:
    0: SELECT_UNIVERSE
    1: WITNESS_NEAREST
    2: ATTEND
    3: FFN_STEP
    4: PROJECT_LAYER
    5: RETURN_SET
    6: ACCEPT
    7: REJECT

Tgt layer classes:
    0: PHONEME
    1: SYLLABLE
    2: MORPHEME
    3: WORD
    4: PHRASE
    5: SEMANTIC
    6: DISCOURSE
    7: none
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

FEATURE_DIM  = 25
N_OP_CLASSES = 8
N_TGT_CLASSES = 8

OP_NAMES = [
    "SELECT_UNIVERSE", "WITNESS_NEAREST", "ATTEND", "FFN_STEP",
    "PROJECT_LAYER",   "RETURN_SET",      "ACCEPT", "REJECT",
]

TGT_NAMES = [
    "PHONEME", "SYLLABLE", "MORPHEME", "WORD",
    "PHRASE",  "SEMANTIC", "DISCOURSE", "none",
]


# ── ProposerModel ──────────────────────────────────────────────────────────────

class ProposerModel(nn.Module):
    """
    Two-layer MLP with dual heads for op_kind and tgt_layer prediction.
    Designed to be small enough to export to ONNX and run from Rust via ort.
    """

    def __init__(
        self,
        input_dim:   int = FEATURE_DIM,
        hidden_dim:  int = 64,
        n_op:        int = N_OP_CLASSES,
        n_tgt:       int = N_TGT_CLASSES,
        dropout:     float = 0.1,
    ):
        super().__init__()
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.n_op       = n_op
        self.n_tgt      = n_tgt

        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.op_head  = nn.Linear(hidden_dim, n_op)
        self.tgt_head = nn.Linear(hidden_dim, n_tgt)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [N, 32] float32 feature vectors
        Returns:
            op_logits:  [N, 8]
            tgt_logits: [N, 8]
        """
        h = self.shared(x)
        return self.op_head(h), self.tgt_head(h)

    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns argmax predictions (no gradient).
        Returns:
            op_pred:  [N] int64
            tgt_pred: [N] int64
        """
        self.eval()
        with torch.no_grad():
            op_logits, tgt_logits = self(x)
            return op_logits.argmax(dim=1), tgt_logits.argmax(dim=1)

    def predict_distribution(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns softmax distributions.
        Returns:
            op_probs:  [N, 8]
            tgt_probs: [N, 8]
        """
        self.eval()
        with torch.no_grad():
            op_logits, tgt_logits = self(x)
            return F.softmax(op_logits, dim=1), F.softmax(tgt_logits, dim=1)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ── ProposerLoss ───────────────────────────────────────────────────────────────

class ProposerLoss(nn.Module):
    """
    Combined loss for op_kind (weighted) and tgt_layer prediction.
    L = CrossEntropy(op_kind, weight=class_weights) + 0.5 * CrossEntropy(tgt_layer)
    """

    def __init__(self, class_weights: Optional[torch.Tensor] = None, tgt_weight: float = 0.5):
        super().__init__()
        self.tgt_weight = tgt_weight
        self.op_loss_fn  = nn.CrossEntropyLoss(weight=class_weights)
        self.tgt_loss_fn = nn.CrossEntropyLoss(ignore_index=7)  # ignore tgt=none

    def forward(
        self,
        op_logits:  torch.Tensor,   # [N, 8]
        tgt_logits: torch.Tensor,   # [N, 8]
        op_labels:  torch.Tensor,   # [N] int64
        tgt_labels: torch.Tensor,   # [N] int64
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            total_loss, op_loss, tgt_loss
        """
        op_loss  = self.op_loss_fn(op_logits, op_labels)
        tgt_loss = self.tgt_loss_fn(tgt_logits, tgt_labels)
        total    = op_loss + self.tgt_weight * tgt_loss
        return total, op_loss, tgt_loss


# ── Accuracy helpers ───────────────────────────────────────────────────────────

def op_accuracy(op_logits: torch.Tensor, op_labels: torch.Tensor) -> float:
    preds = op_logits.argmax(dim=1)
    return (preds == op_labels).float().mean().item()


def tgt_accuracy(tgt_logits: torch.Tensor, tgt_labels: torch.Tensor) -> float:
    mask  = tgt_labels != 7  # exclude none
    if mask.sum() == 0:
        return 1.0
    preds = tgt_logits.argmax(dim=1)
    return (preds[mask] == tgt_labels[mask]).float().mean().item()


def per_class_accuracy(
    op_logits: torch.Tensor,
    op_labels: torch.Tensor,
) -> dict:
    preds = op_logits.argmax(dim=1)
    result = {}
    for i, name in enumerate(OP_NAMES):
        mask = op_labels == i
        if mask.sum() == 0:
            result[name] = None
        else:
            result[name] = (preds[mask] == i).float().mean().item()
    return result


# ── Model I/O ──────────────────────────────────────────────────────────────────

def save_model(model: ProposerModel, path: str):
    torch.save({
        "state_dict":  model.state_dict(),
        "input_dim":   model.input_dim,
        "hidden_dim":  model.hidden_dim,
        "n_op":        model.n_op,
        "n_tgt":       model.n_tgt,
        "n_params":    model.n_params(),
    }, path)
    print(f"Saved model to {path} ({model.n_params()} params)")


def load_model(path: str) -> ProposerModel:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = ProposerModel(
        input_dim  = ckpt["input_dim"],
        hidden_dim = ckpt["hidden_dim"],
        n_op       = ckpt["n_op"],
        n_tgt      = ckpt["n_tgt"],
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"Loaded model from {path} ({ckpt['n_params']} params)")
    return model


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("ProposerModel smoke test")
    model = ProposerModel()
    print(f"  Parameters: {model.n_params()}")
    print(f"  Input dim:  {model.input_dim}")
    print(f"  Op classes: {model.n_op}")
    print(f"  Tgt classes:{model.n_tgt}")

    # Random batch
    x = torch.randn(16, FEATURE_DIM)
    op_logits, tgt_logits = model(x)
    print(f"  op_logits shape:  {tuple(op_logits.shape)}")
    print(f"  tgt_logits shape: {tuple(tgt_logits.shape)}")

    # Predictions
    op_pred, tgt_pred = model.predict(x)
    print(f"  op_pred:  {op_pred.tolist()}")
    print(f"  tgt_pred: {tgt_pred.tolist()}")

    # Loss
    loss_fn  = ProposerLoss()
    op_labels  = torch.randint(0, 8, (16,))
    tgt_labels = torch.randint(0, 8, (16,))
    total, op_loss, tgt_loss = loss_fn(op_logits, tgt_logits, op_labels, tgt_labels)
    print(f"  total_loss={total.item():.4f}  op_loss={op_loss.item():.4f}  tgt_loss={tgt_loss.item():.4f}")

    # Accuracy
    acc = op_accuracy(op_logits, op_labels)
    print(f"  op_accuracy (random baseline): {acc:.3f} (expect ~0.125)")

    print("  OK")
