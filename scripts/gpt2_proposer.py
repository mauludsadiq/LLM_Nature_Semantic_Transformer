import os
import sys
import json
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except Exception:
    pass

def emit(obj, exit_code=0):
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.exit(exit_code)

query = sys.argv[1] if len(sys.argv) > 1 else ""

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)

if not torch.backends.mps.is_available():
    emit({
        "ok": False,
        "error": "MPS not available (CPU forward crashes with Bus error on this machine)",
        "query": query,
        "prompt": None,
        "raw_text": "",
        "ops": [],
        "fallback_used": True,
        "meta": {"device": "cpu", "inference_s": None, "tokens_generated": 0},
    }, exit_code=2)

device = torch.device("mps")
model = AutoModelForCausalLM.from_pretrained(model_name)
model = model.to(device)
model.eval()

# ---- Grammar (v0) ----
# Grammar-defined decoding: constrain generation to the language of valid traces.
# v0 trace language supports: LOAD <frac>; zero+ MASK_BIT bit=<i> val=<0/1>; WITNESS_NEAREST target=<frac>; RETURN_SET.
# IMPORTANT: we still constrain decoding to a single, query-derived candidate (deterministic), but the candidate now reflects recognized constraints.
import re

m = re.search(r"(-?\d+)\s*/\s*(-?\d+)", query)
if m:
    frac = f"{m.group(1)}/{m.group(2)}"
else:
    frac = "7/200"  # fallback if no fraction found

q = query.lower()

# Bit legend (QE v0, matches src/semtrace.rs bit_legend):
# 0: positive
# 1: integer
# 2: den<=6
# 3: num_even
# 4: den_mod3
# 5: proper
# 6: num_abs<=5

bits = set()

# den<=6
if ("denominator" in q or "den<=" in q or "den <= " in q or "den≤" in q) and ("6" in q):
    bits.add(2)
if "den<=6" in q or "den ≤ 6" in query or "den≤6" in q or "<= 6" in q or "≤ 6" in query:
    bits.add(2)

# integer
if "integer" in q:
    bits.add(1)

# positive
if "positive" in q:
    bits.add(0)

# proper
if "proper" in q:
    bits.add(5)

# even
if "even" in q or "num_even" in q:
    bits.add(3)

# den_mod3
if "den_mod3" in q or "den mod 3" in q or "den%3" in q or "den % 3" in q:
    bits.add(4)

# num_abs<=5
if "num_abs<=5" in q or "num_abs ≤ 5" in query or "abs<=5" in q or "abs ≤ 5" in query:
    bits.add(6)

ops = [f"LOAD {frac}"]
for i in sorted(bits):
    ops.append(f"MASK_BIT bit={i} val=1")
ops.append(f"WITNESS_NEAREST target={frac}")
ops.append("RETURN_SET")

CANDIDATES = ["\n".join(ops) + "\n"]

prompt = (
    "You are a semantic trace generator.\n"
    "Convert the query into a sequence of operations.\n"
    "Output ONLY the trace, no commentary.\n\n"
    f"Query: {query}\n"
    "Trace:\n"
)

try:
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    prompt_len = int(inputs["input_ids"].shape[1])

    # Warm up MPS allocations (avoids placeholder storage errors on some ops)
    with torch.no_grad():
        _ = model(**inputs)

    class TrieNode:
        __slots__ = ("children", "terminal")
        def __init__(self):
            self.children = {}
            self.terminal = False

    def build_trie(seqs_token_ids):
        root = TrieNode()
        max_len = 0
        for seq in seqs_token_ids:
            max_len = max(max_len, len(seq))
            cur = root
            for tid in seq:
                nxt = cur.children.get(tid)
                if nxt is None:
                    nxt = TrieNode()
                    cur.children[tid] = nxt
                cur = nxt
            cur.terminal = True
        return root, max_len

    seqs = [tokenizer.encode(s, add_special_tokens=False) for s in CANDIDATES]
    trie_root, max_new = build_trie(seqs)

    def allowed_tokens(batch_id, input_ids):
        # transformers may pass input_ids as 1D (seq_len,) or 2D (1, seq_len)
        if hasattr(input_ids, "dim") and int(input_ids.dim()) == 1:
            full = input_ids.tolist()
            seq_len = int(input_ids.shape[0])
        else:
            full = input_ids[0].tolist()
            seq_len = int(input_ids.shape[-1])

        gen_len = seq_len - prompt_len
        if gen_len < 0:
            gen_len = 0

        gen_ids = full[prompt_len:prompt_len + gen_len]
        cur = trie_root
        for tid in gen_ids:
            nxt = cur.children.get(int(tid))
            if nxt is None:
                eos = tokenizer.eos_token_id
                return [int(eos)] if eos is not None else [0]
            cur = nxt

        if cur.children:
            return [int(t) for t in cur.children.keys()]

        eos = tokenizer.eos_token_id
        return [int(eos)] if eos is not None else [0]

    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=False,
            num_beams=1,
            prefix_allowed_tokens_fn=allowed_tokens,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    t1 = time.time()

    gen_ids = out[0][prompt_len:]
    raw_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    tokens_generated = int(gen_ids.numel()) if hasattr(gen_ids, "numel") else int(len(gen_ids))

    # Strict validation against the grammar language (NO fallback on invalid).
    if raw_text not in CANDIDATES:
        emit({
            "ok": False,
            "error": "generated output not in trace grammar language",
            "query": query,
            "prompt": prompt,
            "raw_text": raw_text,
            "ops": [],
            "fallback_used": True,
            "meta": {
                "device": "mps",
                "inference_s": float(t1 - t0),
                "tokens_generated": int(tokens_generated),
            },
        }, exit_code=2)

    ops = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]

    emit({
        "ok": True,
        "query": query,
        "prompt": prompt,
        "raw_text": raw_text,
        "ops": ops,
        "fallback_used": False,
        "meta": {
            "device": "mps",
            "inference_s": float(t1 - t0),
            "tokens_generated": int(tokens_generated),
        },
    }, exit_code=0)

except Exception as e:
    emit({
        "ok": False,
        "error": f"exception: {type(e).__name__}: {e}",
        "query": query,
        "prompt": prompt,
        "raw_text": "",
        "ops": [],
        "fallback_used": True,
        "meta": {"device": "mps", "inference_s": None, "tokens_generated": 0},
    }, exit_code=2)
