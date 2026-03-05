"""
ONNX export — Stage 14.
Usage: python train/export.py --model train/model_v2.pt --out train/model.onnx
"""

import argparse, os
import torch
import torch.nn as nn

FDIM   = 25
HIDDEN = 128
N_OPS  = 8

OP_NAMES = ["SELECT_UNIVERSE","WITNESS_NEAREST","ATTEND","FFN_STEP",
            "PROJECT_LAYER","RETURN_SET","ACCEPT","REJECT"]


def build_model():
    shared = nn.Sequential(
        nn.Linear(FDIM, HIDDEN), nn.ReLU(), nn.Dropout(0.0),
        nn.Linear(HIDDEN, HIDDEN), nn.ReLU(), nn.Dropout(0.0),
    )
    return shared, nn.Linear(HIDDEN, N_OPS), nn.Linear(HIDDEN, N_OPS)


class ProposerOnnx(nn.Module):
    def __init__(self, shared, op_head, tgt_head):
        super().__init__()
        self.shared = shared
        self.op_head = op_head
        self.tgt_head = tgt_head

    def forward(self, x: torch.Tensor):
        h = self.shared(x)
        return self.op_head(h), self.tgt_head(h)


def export(model_path: str, out_path: str):
    print(f"Loading {model_path}...")
    ckpt = torch.load(model_path, map_location="cpu")
    shared, op_head, tgt_head = build_model()
    shared.load_state_dict(ckpt["shared"])
    op_head.load_state_dict(ckpt["op_head"])
    tgt_head.load_state_dict(ckpt["tgt_head"])
    model = ProposerOnnx(shared, op_head, tgt_head)
    model.eval()

    print(f"Exporting to {out_path}...")
    torch.onnx.export(
        model,
        torch.zeros(1, FDIM),
        out_path,
        input_names=["features"],
        output_names=["op_logits", "tgt_logits"],
        dynamic_axes={
            "features":   {0: "batch_size"},
            "op_logits":  {0: "batch_size"},
            "tgt_logits": {0: "batch_size"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    print("Exported.")

    # Verify
    import onnxruntime as ort
    import numpy as np
    sess = ort.InferenceSession(out_path)

    # Test all 12 block positions with correct (block_idx, active_layer) pairs
    # Layer advances as FFN_STEP traverses the tower upward
    print("\nVerification (block_idx + correct layer → expected op):")
    expected = {
        0:  ("SELECT_UNIVERSE", 0),   # PHONEME
        1:  ("WITNESS_NEAREST", 0),   # PHONEME
        2:  ("ATTEND",          0),   # PHONEME
        3:  ("FFN_STEP",        0),   # PHONEME→SYLLABLE
        4:  ("FFN_STEP",        1),   # SYLLABLE→MORPHEME
        5:  ("FFN_STEP",        2),   # MORPHEME→WORD
        6:  ("FFN_STEP",        3),   # WORD→PHRASE
        7:  ("FFN_STEP",        4),   # PHRASE→SEMANTIC
        8:  ("FFN_STEP",        5),   # SEMANTIC→DISCOURSE
        9:  ("PROJECT_LAYER",   6),   # DISCOURSE
        10: ("RETURN_SET",      6),   # DISCOURSE
        11: ("ACCEPT",          6),   # DISCOURSE
    }
    all_pass = True
    for bi, (exp_op, layer_idx) in expected.items():
        x_np = np.zeros((1, FDIM), dtype=np.float32)
        x_np[0, bi] = 1.0                  # block_idx one-hot [0..11]
        x_np[0, 12 + layer_idx] = 1.0      # layer one-hot [12..18]
        x_np[0, 21] = 1.0                  # tau=1.0
        x_np[0, 24] = 0.3                  # top_k=3
        x_np[0, 19] = min(bi / 20.0, 1.0)  # step_count
        op_logits, _ = sess.run(None, {"features": x_np})
        pred = OP_NAMES[op_logits.argmax(axis=1)[0]]
        ok = "✓" if pred == exp_op else "✗"
        if pred != exp_op: all_pass = False
        print(f"  block={bi:2d} layer={layer_idx}  pred={pred:<20} expected={exp_op:<20} {ok}")

    print(f"\nONNX verification: {'✓ PASSED' if all_pass else '✗ FAILED'}")
    print(f"model.onnx: {os.path.getsize(out_path)} bytes")
    return all_pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="train/model_v2.pt")
    parser.add_argument("--out",   default="train/model.onnx")
    args = parser.parse_args()
    export(args.model, args.out)
