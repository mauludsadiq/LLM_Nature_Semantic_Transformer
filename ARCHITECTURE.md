# Architecture Reference

## LLM Nature — Semantic Transformer

## Overview

The system is a Proposer / Executor / Verifier architecture over a certified 7-layer linguistic universe.

Weights do not store the world. The tower stores the world. Weights learn to navigate it.

This eliminates hallucination at the architectural level. The proposer emits ops. The executor runs them deterministically. The verifier certifies or rejects. No output is accepted without a valid cryptographic digest chain.

---

## Two-Phase Build

Phase 1 (Stages 0-9, Rust): Certified tower — 7-layer linguistic universe with cryptographic identity, certified attention/FFN/transformer, rule-based proposer. 317 tests passing.

Phase 2 (Stages 10-17, Python): Learned proposer — neural net trained on verified tower traces via imitation learning, REINFORCE, and process supervision. Exported to ONNX (88KB).

---

## System Components

QUERY
  -> PROPOSER (learned)
       Input:  ProposerContext (active_layer, step_count, chain_hash, tau, top_k)
       Output: OpDistribution over 8 op classes
       Model:  Linear(25->128) ReLU Linear(128->128) ReLU -> op_head / tgt_head
       Params: 7,440
  -> EXECUTOR (deterministic)
       Runs op against certified tower
       Produces: StepRec { pre, post, step_digest }
       step_digest = SHA-256(chain || op || args || set_digest)
  -> VERIFIER (deterministic, independent)
       Replays every step from NDJSON trace
       Checks step_digest matches for every record
       Result: Accept | Reject (no partial credit)

---

## The Tower: 7 Certified Layers

DISCOURSE (L7)  5 certified graphs, 16-bit sig
  depends on
SEMANTIC  (L6)  6 certified graphs, 16-bit sig
  depends on
PHRASE    (L5)  5 certified parse trees, 16-bit sig
  depends on
WORD      (L4)  34,487 words, 8-bit sig (CMU + WordNet)
  depends on
MORPHEME  (L3)  16 morphemes, 16-bit sig
  depends on
SYLLABLE  (L2)  ~423,000 syllables, 16-bit sig
  depends on
PHONEME   (L1)  44 phonemes, 12-bit sig

Every element has: canonical byte encoding, structural signature, SHA-256 digest.
Universe digest = Merkle root over element digests, bound to validation rules.
Any mutation propagates immediately to tower_root and invalidates all downstream traces.

---

## Cryptographic Identity

element_digest  = SHA-256(canonical_bytes(element))
universe_digest = SHA-256(
    name_tag
    || SHA-256(validation_rules)
    || SHA-256(signature_bit_legend)
    || merkle_root(sorted element_digests)
)
tower_root = SHA-256(sorted universe_digests)

---

## Signature Encoding

Layer      Bits  Encodes
PHONEME      12  manner(4) place(4) voicing(1) vowel(1) stress(1) length(1)
SYLLABLE     16  onset_complexity(3) nucleus_type(3) coda_complexity(3) stress(2) length(2) sonority(3)
MORPHEME     16  category(4) allomorphy_class(4) meaning_class(4) bound(1) productive(1) inflectional(1) derivational(1)
WORD          8  pos(3) syllable_count(2) stress_pattern(2) frequency_class(1)
PHRASE       16  node_types_present(10) max_depth(3) branching(3)
SEMANTIC     16  relation_types_present(10) entity_count(2) event_count(2) polarity(1) tense(1)
DISCOURSE    16  relation_types_present(8) graph_count(3) coref_present(1) negation(1) resolved(1) unknown(1) temporal_order(1)

---

## Certified Tower Components

CertifiedAttention (attention.rs)
  score(q, k) = exp(-hamming(sig(q), sig(k)) / tau)
  attn(q, K)  = softmax(scores) * values
  No learned parameters. Distances are exact integers. Numerically stable via log-sum-exp.

TowerFFN (feedforward.rs)
  FFN_STEP(x, layer_from, layer_to):
    edge       = lookup_edge(layer_from, layer_to)
    candidates = edge.project(x)
    witness    = nearest(candidates, x.sig)   // Hamming nearest
    return (witness, step_digest)
  6 certified edges: PHONEME->SYLLABLE->MORPHEME->WORD->PHRASE->SEMANTIC->DISCOURSE

TowerTransformer (transformer.rs)
  One block = one upward pass through all 6 edges.
  chain = SHA-256(chain || step_digest) at each edge.
  chain is the cryptographic commitment to the entire forward pass.

SigIndex (sig_index.rs)
  Certified inverted index: sig -> Vec<ElementId>
  Nearest lookup: iterate over all 65,536 sigs in order of Hamming distance from query.
  Tamper detection: recompute posting digest and compare.

---

## The Learned Proposer

ProposerContext (runtime state at each step)
  active_layer, step_count, rejection_count,
  pass_digest, witness_digest, chain_hash, tau, top_k

FeatureEncoder (25 dimensions)
  [0..11]   block_idx one-hot (12 dims)   -- position in canonical sequence
  [12..18]  active_layer one-hot (7 dims)
  [19]      step_count / 20.0
  [20..23]  tau bin one-hot: [<=0.5, <=1.0, <=2.0, 4.0+]
  [24]      top_k / 10.0
  All values in [0.0, 1.0]. Deterministic and platform-independent.

ProposerModel
  Linear(25->128) + ReLU + Dropout(0.1)
  Linear(128->128) + ReLU + Dropout(0.1)
  op_head:  Linear(128->8)   op_kind logits
  tgt_head: Linear(128->8)   tgt_layer logits
  Total: 7,440 parameters
  ONNX: train/model.onnx (88,804 bytes)

---

## Training Pipeline

Stage 13 -- Imitation Learning
  Loss: CrossEntropy(op_kind, class_weights) + 0.5 * CrossEntropy(tgt_layer)
  Class weights: inverse-frequency normalized (mean=1.0)
  REJECT class weight: 15.211 (35 samples vs 2112 FFN_STEP)
  Result: 100% val accuracy on all 8 op classes

Stage 16 -- REINFORCE
  Conservative: update only when model errs on noisy inputs
  loss = -log_prob(gt_action) + 0.1 * KL(policy || reference)
  Reference policy: frozen IL checkpoint. lr=5e-5
  Result: 1.000 clean and noisy acceptance over 20 epochs

Stage 17 -- Process Supervision
  Step-level rewards (not sparse terminal):
    correct:   +1.0  (+2.0 bonus at terminal block 11)
    wrong:     -0.5  (-1.0 penalty at terminal block 11)
  Discounted return: G_t = r_t + 0.99*r_{t+1} + ...
  Curriculum noise: 0 -> 0.15 over 10 epochs, held at 0.15
  Result: 1.000 step_acc and pass_acc clean+noisy over 30 epochs

---

## Canonical Op Sequence

block  op               active_layer
  0    SELECT_UNIVERSE  PHONEME
  1    WITNESS_NEAREST  PHONEME
  2    ATTEND           PHONEME
  3    FFN_STEP         PHONEME -> SYLLABLE
  4    FFN_STEP         SYLLABLE -> MORPHEME
  5    FFN_STEP         MORPHEME -> WORD
  6    FFN_STEP         WORD -> PHRASE
  7    FFN_STEP         PHRASE -> SEMANTIC
  8    FFN_STEP         SEMANTIC -> DISCOURSE
  9    PROJECT_LAYER    DISCOURSE
  10   RETURN_SET       DISCOURSE
  11   ACCEPT           DISCOURSE

---

## Op Class Encoding

class  op               typical block
  0    SELECT_UNIVERSE  0
  1    WITNESS_NEAREST  1
  2    ATTEND           2
  3    FFN_STEP         3-8
  4    PROJECT_LAYER    9
  5    RETURN_SET       10
  6    ACCEPT           11
  7    REJECT           synthetic (every 10th pass)

---

## Data Formats

corpus_v2.ndjson -- 4259 records, op entropy 2.342 bits
  { pass_idx, block_idx, op_kind, op_class, active_layer, tgt_layer,
    tgt_class, step_count, rejection_count, tau, top_k, chain_hash,
    pass_digest, phoneme_idx }

features.bin -- [N x 25] f32 little-endian
labels.bin   -- [N x 2] u8 pairs (op_class, tgt_class)
splits.json  -- { n_train, n_val, n_test, class_weights[8] }
              -- split: 80% train / 10% val / 10% test (LCG shuffle seed=42)

---

## Model Files

train/model_v2.pt   IL checkpoint (100% val acc)
train/model_rl.pt   After REINFORCE (1.000 clean+noisy)
train/model_ps.pt   After process supervision (final)
train/model.onnx    ONNX export (88,804 bytes)

---

## ONNX Integration (Rust via ort)

[dependencies]
ort = { version = "2", features = ["load-dynamic"] }

let env     = Environment::builder().build()?;
let session = SessionBuilder::new(&env)?.with_model_from_file("train/model.onnx")?;
let features: Array2<f32> = encode_context(&ctx, tau, top_k, block_idx);
let outputs  = session.run(inputs![features])?;
let op_logits: ArrayView2<f32> = outputs[0].try_extract()?;
let op_class = op_logits.row(0).argmax().unwrap();

The ONNX model is a drop-in replacement for RuleBasedProposer.
Interface: context features in, op distribution out.

---

## Notes on Determinism

- No floating point used for ordering or distance in the tower
- Nearest-witness: exact integer Hamming distance over signatures
- Canonical byte encodings sort lexicographically -- same result on any platform
- Feature vectors: same context -> same 25-dim vector on any platform
- ONNX inference: reproducible across platforms
