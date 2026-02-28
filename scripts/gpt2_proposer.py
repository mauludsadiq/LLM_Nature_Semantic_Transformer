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
# Deterministic, grammar-defined decoding: constrain GPT-2 to emit EXACTLY ONE
# candidate trace string derived from the query.
#
# v0 supported trace languages:
#   QE: LOAD <frac>; zero+ MASK_BIT bit=<i> val=<0/1>; WITNESS_NEAREST target=<frac>; RETURN_SET
#   GE: START_ELEM <a,b,c>; zero+ SET_BIT i=<i> b=<0/1>; WITNESS_NEAREST target_elem=<a,b,c> metric=ABS_DIFF; RETURN_SET max_items=<n> include_witness=<bool>
import re

q = query.lower()

# Detect GE intent.
is_ge = ("in ge" in q) or ("universe\":\"ge\"" in q) or ("triangle" in q) or (re.search(r"\b\d+\s*,\s*\d+\s*,\s*\d+\b", query) is not None)

# Extract triangle if present.
mt = re.search(r"\b(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\b", query)
tri = None
if mt:
    tri = f"{mt.group(1)},{mt.group(2)},{mt.group(3)}"

# Extract fraction if present.
mf = re.search(r"(-?\d+)\s*/\s*(-?\d+)", query)
if mf:
    frac = f"{mf.group(1)}/{mf.group(2)}"
else:
    frac = "7/200"  # fallback if no fraction found

# Parse RETURN_SET presentation hints (deterministic, best-effort).
max_items = 20
mm = re.search(r"\b(return\s+up\s+to|up\s+to|max[_ ]items)\s+(\d+)\b", q)
if mm:
    try:
        max_items = int(mm.group(2))
    except Exception:
        max_items = 20
max_items = max(1, min(max_items, 200))

include_witness = ("include witness" in q) or ("include_witness" in q)

if is_ge:
    # GE bit legend (matches src/semtrace.rs bit_legend_geom):
    # 0: perim<=20
    # 1: isosceles
    # 2: equilateral
    # 3: primitive
    # 4: right
    # 5: acute
    # 6: obtuse
    bits = []

    if ("perimeter" in q or "perim" in q) and ("<= 20" in q or "≤ 20" in query or "perim<=20" in q):
        bits.append((0, 1))
    if "primitive" in q:
        bits.append((3, 1))
    if "right" in q or "right-angled" in q or "right angled" in q:
        bits.append((4, 1))

    if tri is None:
        tri = "5,12,13"  # deterministic fallback target

    ops = [f"START_ELEM {tri}"]
    for (i, b) in bits:
        ops.append(f"SET_BIT i={i} b={b}")
    ops.append(f"WITNESS_NEAREST target_elem={tri} metric=ABS_DIFF")
    include_witness_val = "true" if include_witness else "false"
    ops.append(f"RETURN_SET max_items={max_items} include_witness={include_witness_val}")
else:
    # QE bit legend (matches src/semtrace.rs bit_legend):
    # 0: positive
    # 1: integer
    # 2: den<=6
    # 3: num_even
    # 4: den_mod3
    # 5: proper
    # 6: num_abs<=5
    bits = set()
    if ("denominator" in q or "den<=" in q or "den <= " in q or "den≤" in q) and ("6" in q):
        bits.add(2)
    if "den<=6" in q or "den ≤ 6" in query or "den≤6" in q or "<= 6" in q or "≤ 6" in query:
        bits.add(2)
    if "integer" in q:
        bits.add(1)
    if "positive" in q:
        bits.add(0)
    if "proper" in q:
        bits.add(5)
    if "even" in q or "num_even" in q:
        bits.add(3)
    if "den_mod3" in q or "den mod 3" in q or "den%3" in q or "den % 3" in q:
        bits.add(4)
    if "num_abs<=5" in q or "num_abs ≤ 5" in query or "abs<=5" in q or "abs ≤ 5" in query:
        bits.add(6)

    ops = [f"LOAD {frac}"]
    for i in sorted(bits):
        ops.append(f"MASK_BIT bit={i} val=1")
    ops.append(f"WITNESS_NEAREST target={frac}")
    ops.append("RETURN_SET")

# k=2 candidates: ordering variation only (same ops, different bit-application order), then deterministic collapse to 1
# This preserves: same query -> same chosen trace string.
if is_ge:
    # ops = [START_ELEM] + SET_BIT* + [WITNESS_NEAREST, RETURN_SET]
    # reorder only the SET_BIT lines
    bit_lines = [ln for ln in ops if ln.startswith("SET_BIT ")]
    tail = [ln for ln in ops if not ln.startswith("SET_BIT ") and not ln.startswith("START_ELEM ")]
    head = [ops[0]]
    ops_alt = head + list(reversed(bit_lines)) + tail
else:
    # QE: reorder only the MASK_BIT lines
    bit_lines = [ln for ln in ops if ln.startswith("MASK_BIT ")]
    tail = [ln for ln in ops if not ln.startswith("MASK_BIT ") and not ln.startswith("LOAD ")]
    head = [ops[0]]
    ops_alt = head + list(reversed(bit_lines)) + tail

cand_a = "\n".join(ops) + "\n"
cand_b = "\n".join(ops_alt) + "\n"
cands = [cand_a, cand_b]
cands.sort()
CANDIDATES = cands

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
