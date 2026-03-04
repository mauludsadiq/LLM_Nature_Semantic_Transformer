# The Collapse Principle

## Core Statement

In this stack, every "infinite" in language is never represented as an unbounded set.
It is always represented as a finite, certified universe plus finite operators that map
raw input into that universe.

The system collapses open-endedness by forcing every representation to be:

1. **Enumerable** — finite inventory or finite bounds
2. **Canonical** — exactly one encoding per object
3. **Typed** — schema + validators
4. **Digest-locked** — `universe_digest` + `trace_chain_hash`

The only way "infinite" shows up is as many possible inputs hitting the system over time.
Each run collapses to a finite artifact.

---

## What "Infinite" Means in Language, and How We Collapse Each One

### Infinite 1: Infinite audio waveforms

**Reality:** infinitely many acoustic realizations of the "same" sound.

**Collapse:** audio → phoneme ID via a deterministic front-end + certified inventory.
- The phoneme universe is finite (inventory file with stable IDs).
- Canonicalization guarantees one encoding per phoneme + feature bundle.
- Output is not "a sound" — it is a finite ID + finite feature bits.

**Result:** infinite waveform variability collapses into a finite phoneme universe.

---

### Infinite 2: Infinite pronunciations / coarticulation / speaker variation

**Reality:** pronunciation changes by speaker, context, speed.

**Collapse:** pronunciation differences are treated as non-identity; identity is the
canonical unit.
- If two realizations map to the same certified phoneme, they become the same object.
- Anything else fails validation or becomes a different object.

**Result:** variation becomes either (a) the same canonical unit or (b) a different unit.
No infinite fuzz.

---

### Infinite 3: Infinite strings of sounds (arbitrary length)

**Reality:** you can always add another phoneme.

**Collapse:** the universe is not "all possible strings." The universe is:
- finite syllable inventory (bounded length rules)
- finite morpheme inventory (meaning IDs + allowed allomorph sets)
- finite word inventory (morpheme sequences constrained by word_rules)
- finite phrase inventory (parse trees constrained by grammar + node limits)
- finite semantic graphs (bounded nodes/edges)
- finite discourse graphs (max_nodes/max_edges)

**Key move:** bounded construction at every layer.

**Result:** "infinite length language" collapses because each universe enforces maximum
size and only admits certified objects.

---

### Infinite 4: Infinite paraphrases (same meaning, different surface)

**Reality:** many sentences mean the same thing.

**Collapse:** meaning identity lives at the semantic graph layer.
- Different phrase trees can map to the same semantic graph.
- Canonical semantic graph encoding makes "same meaning" literally identical bytes →
  identical digest.

**Result:** paraphrase infinity collapses to a finite set of canonical semantic graphs.

---

### Infinite 5: Infinite discourse continuations

**Reality:** discourse is unbounded over time.

**Collapse (Layer 7):** any single discourse artifact is finite by construction:
- `max_nodes=512`, `max_edges=1024`
- nodes are semantic_graph IDs (finite set per run)
- edges are from a finite `relation_type` enum

**Result:** an "infinite story" becomes a chain of finite discourse graphs, each
digest-locked.

---

### Infinite 6: Infinite ambiguity (pronouns, reference, missing context)

**Reality:** "it" could refer to many things.

**Collapse:** ambiguity is not allowed to remain vague inside an artifact.
- Coreference is a concrete edge between specific entity nodes.
- If resolution cannot be certified, the graph must encode that as a typed
  `truth_status:"unknown"` state — not an open interpretive cloud.

**Result:** ambiguity must become either (a) a specific resolved edge, or (b) a typed
unknown state.

---

## The Key Mechanism: Universe + Canon + Digest

### 1. Universe spec makes the space finite
Inventory + schema + max bounds = finite.

### 2. Canonicalization makes identity single-valued
Exactly one encoding per object. Without canon, the digest system breaks — two
representations of the same object produce different hashes and the system cannot
recognize identity. Canonicalization is what makes "same meaning = same bytes = same
digest" hold. The key-sorting, no-whitespace, and explicit-null rules at each layer
are not style choices — they are correctness requirements.

### 3. Predicate signatures collapse detail into finite classes
Signatures are finite bitsets — a deterministic compression: many objects → one class
label. The selectivity of a constraint over a universe is a constructive probability
measure derived from structure, not assigned from a corpus.

### 4. Trace + chain hash collapses execution into a finite proof object
A run produces:
- `trace.ndjson` — finite list of steps
- `result.json` — finite summary
- `universe_digest / legend_digest / rules_digest / inventory_digest`
- `trace_chain_hash` — one final integrity anchor

Even if input is open-ended, the output is a finite, replayable, auditable artifact.

---

## One Sentence

Language is infinite at the input boundary, but finite at every certified boundary:
each layer only admits finite, canonical, digest-locked objects, and Layer 7 turns
unbounded discourse into a sequence of finite, verifiable knowledge-graph snapshots.
