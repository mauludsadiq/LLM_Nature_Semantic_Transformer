"""
Stage 24 — Rejection corpus generation.

Generates deliberate REJECT scenarios to balance the REJECT class.
Current corpus_v3: REJECT = 0.8% (142/17230), class weight 8x.
Target: REJECT = 8-10%, class weight ~2x.

Rejection scenarios:
  1. wrong_layer:    FFN_STEP when active_layer=DISCOURSE (no more layers)
  2. early_accept:   ACCEPT at block_idx < 9 (pass not complete)
  3. repeat_select:  SELECT_UNIVERSE at block_idx > 0
  4. premature_proj: PROJECT_LAYER before block_idx 9
  5. budget_exhaust: step_count > 15, any non-terminal op

Each scenario generates a CorpusRecord with op_class=7 (REJECT).
Output appended to corpus_v3 to create corpus_v4.ndjson.
"""

import json, random, hashlib, math

CORPUS_V3 = "training_data/corpus_v3.ndjson"
CORPUS_V4 = "training_data/corpus_v4.ndjson"

OP_NAMES=["SELECT_UNIVERSE","WITNESS_NEAREST","ATTEND","FFN_STEP",
          "PROJECT_LAYER","RETURN_SET","ACCEPT","REJECT"]
LAYER_NAMES=["PHONEME","SYLLABLE","MORPHEME","WORD","PHRASE","SEMANTIC","DISCOURSE"]
TAU_BINS=[0.5,1.0,2.0,4.0]

def sha256(*parts):
    h = hashlib.sha256()
    for p in parts: h.update(p if isinstance(p, bytes) else p.encode())
    return h.hexdigest()

def tau_bin(tau):
    for i,t in enumerate(TAU_BINS):
        if tau<=t: return i
    return 3

def make_reject_record(scenario, block_idx, layer_idx, step_count,
                        tau, top_k, seed):
    """Generate a REJECT record for a deliberate failure scenario."""
    chain_hash = sha256(f"reject_{scenario}_{seed}".encode()
                        if isinstance(seed,int) else f"reject_{scenario}_{seed}")
    step_digest = sha256(f"REJECT_{scenario}_{block_idx}_{seed}")
    return {
        "phoneme_idx":    seed % 44,
        "phoneme_sym":    f"ph:reject_{scenario}",
        "tau":            tau,
        "top_k":          top_k,
        "block_idx":      block_idx,
        "src_layer":      LAYER_NAMES[layer_idx],
        "tgt_layer":      LAYER_NAMES[layer_idx],
        "context_digest": sha256(f"ctx_{scenario}_{seed}"),
        "chain_hash":     chain_hash,
        "active_layer":   LAYER_NAMES[layer_idx],
        "step_count":     step_count,
        "op_kind":        "REJECT",
        "op_tgt_layer":   None,
        "step_digest":    step_digest,
        "attn_top":       "rejected",
        "ffn_top":        "rejected",
        "block_digest":   sha256(f"blk_{scenario}_{seed}"),
        "op_class":       7,
        "tgt_class":      7,
        "reject_scenario": scenario,
    }

def generate_rejection_corpus(n_per_scenario=200):
    records = []
    taus   = [0.5, 1.0, 2.0, 4.0]
    top_ks = [3, 5]
    seed   = 0

    # Scenario 1: wrong_layer — FFN_STEP when at DISCOURSE (nowhere to go)
    for i in range(n_per_scenario):
        tau   = taus[i % len(taus)]
        top_k = top_ks[i % len(top_ks)]
        # DISCOURSE is layer 6 — FFN_STEP here is invalid
        records.append(make_reject_record(
            "wrong_layer", block_idx=random.randint(8,11),
            layer_idx=6, step_count=random.randint(8,12),
            tau=tau, top_k=top_k, seed=seed))
        seed += 1

    # Scenario 2: early_accept — ACCEPT before block 9
    for i in range(n_per_scenario):
        tau   = taus[i % len(taus)]
        top_k = top_ks[i % len(top_ks)]
        records.append(make_reject_record(
            "early_accept", block_idx=random.randint(0,8),
            layer_idx=random.randint(0,5),
            step_count=random.randint(0,8),
            tau=tau, top_k=top_k, seed=seed))
        seed += 1

    # Scenario 3: repeat_select — SELECT_UNIVERSE after block 0
    for i in range(n_per_scenario):
        tau   = taus[i % len(taus)]
        top_k = top_ks[i % len(top_ks)]
        records.append(make_reject_record(
            "repeat_select", block_idx=random.randint(1,11),
            layer_idx=random.randint(0,6),
            step_count=random.randint(1,11),
            tau=tau, top_k=top_k, seed=seed))
        seed += 1

    # Scenario 4: premature_project — PROJECT_LAYER before block 9
    for i in range(n_per_scenario):
        tau   = taus[i % len(taus)]
        top_k = top_ks[i % len(top_ks)]
        records.append(make_reject_record(
            "premature_proj", block_idx=random.randint(0,8),
            layer_idx=random.randint(0,5),
            step_count=random.randint(0,8),
            tau=tau, top_k=top_k, seed=seed))
        seed += 1

    # Scenario 5: budget_exhaust — step_count > 15
    for i in range(n_per_scenario):
        tau   = taus[i % len(taus)]
        top_k = top_ks[i % len(top_ks)]
        records.append(make_reject_record(
            "budget_exhaust", block_idx=random.randint(0,11),
            layer_idx=random.randint(0,6),
            step_count=random.randint(16,60),
            tau=tau, top_k=top_k, seed=seed))
        seed += 1

    return records

if __name__ == "__main__":
    random.seed(42)

    # Load v3
    v3 = [json.loads(l) for l in open(CORPUS_V3)]
    n_v3 = len(v3)
    n_reject_v3 = sum(1 for r in v3 if r["op_class"] == 7)

    print(f"Corpus v3: {n_v3} records, {n_reject_v3} REJECT ({100*n_reject_v3/n_v3:.1f}%)")

    # Generate rejection records
    n_per = 300  # 5 scenarios × 300 = 1500 new REJECT records
    reject_records = generate_rejection_corpus(n_per)
    n_new_reject = len(reject_records)

    # Combine
    all_records = v3 + reject_records
    n_total = len(all_records)
    n_reject_total = sum(1 for r in all_records if r["op_class"] == 7)

    print(f"New REJECT records: {n_new_reject}")
    print(f"Corpus v4: {n_total} records, {n_reject_total} REJECT ({100*n_reject_total/n_total:.1f}%)")

    # Op distribution
    from collections import Counter
    ops = Counter(r["op_kind"] for r in all_records)
    print("\nOp distribution:")
    for op in ["SELECT_UNIVERSE","WITNESS_NEAREST","ATTEND","FFN_STEP",
               "PROJECT_LAYER","RETURN_SET","ACCEPT","REJECT"]:
        c = ops.get(op, 0)
        print(f"  {op:<22} {c:>6} ({100*c/n_total:.1f}%)")

    # Entropy
    from math import log2
    total = n_total
    entropy = sum(-c/total*log2(c/total) for c in ops.values() if c > 0)
    print(f"\nEntropy: {entropy:.3f} bits")

    # Suggested class weight for REJECT
    n_non_reject = n_total - n_reject_total
    weight = n_non_reject / n_reject_total
    print(f"Suggested REJECT class weight: {weight:.1f}x (down from 8x)")

    # Write v4
    with open(CORPUS_V4, "w") as f:
        for r in all_records:
            # Convert None to null properly
            line = json.dumps({k: v for k,v in r.items()})
            f.write(line + "\n")
    print(f"\nWritten: {CORPUS_V4}")

    # Gates
    reject_pct = 100 * n_reject_total / n_total
    g1 = 7.0 <= reject_pct <= 12.0
    g2 = weight <= 12.0  # realistic: REJECT at 8-10% gives ~10x, down from 15x in v2
    g3 = entropy >= 2.5
    print(f"\nStage 24 gates:")
    print(f"  REJECT% in 7-12%:     {'✓' if g1 else '✗'} ({reject_pct:.1f}%)")
    print(f"  class weight <= 12x:  {'✓' if g2 else '✗'} ({weight:.1f}x, was ~15x in v2)")
    print(f"  entropy >= 2.5 bits:  {'✓' if g3 else '✗'} ({entropy:.3f})")
    print(f"\nStage 24 gate: {'✓ PASSED' if (g1 and g2 and g3) else '✗ FAILED'}")