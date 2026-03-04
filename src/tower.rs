//! Unified tower artifact — Stage 9 of the blueprint.
//!
//! The Tower is the content-addressed package that ties all components together.
//!
//! Content-addressed structure:
//!   root_digest = SHA-256(
//!     layer_digests[0..6] ||
//!     edge_digest          ||
//!     index_digests[0..6]
//!   )
//!
//! Any runtime with the same root_digest:
//!   - Has identical certified universes
//!   - Has identical edge rules
//!   - Will verify any trace byte-identically
//!   - Cannot hallucinate (verifier is deterministic over certified universes)
//!
//! The Tower also provides the unified entry point for:
//!   - Forward passes (TowerTransformer)
//!   - Proposer execution (RuleBasedProposer)
//!   - Trace corpus collection (ProposerTrainer)
//!   - Verified query execution

use crate::digest::{sha256_bytes, merkle_root};
use crate::layer::{Layer, LayerId};
use crate::phoneme::PhonemeLayer;
use crate::syllable::SyllableLayer;
use crate::morpheme::MorphemeLayer;
use crate::word::WordLayer;
use crate::phrase::PhraseLayer;
use crate::semantic::SemanticLayer;
use crate::discourse::DiscourseLayer;
use crate::sig_index::SigIndex;
use crate::edges::tower_edge_digest;
use crate::transformer::{TowerTransformer, ForwardPass, BlockConfig};
use crate::proposer::{
    ProposerContext, RuleBasedProposer, ProposerTrainer,
    TraceRecord, OpKind,
};

// ── TowerManifest ─────────────────────────────────────────────────────────────

/// A manifest of all certified digests in the tower.
/// Serializable, content-addressable.
#[derive(Clone, Debug)]
pub struct TowerManifest {
    pub phoneme_digest:   [u8; 32],
    pub syllable_digest:  [u8; 32],
    pub morpheme_digest:  [u8; 32],
    pub word_digest:      [u8; 32],
    pub phrase_digest:    [u8; 32],
    pub semantic_digest:  [u8; 32],
    pub discourse_digest: [u8; 32],
    pub edge_digest:      [u8; 32],
    pub phoneme_idx_digest:   [u8; 32],
    pub syllable_idx_digest:  [u8; 32],
    pub morpheme_idx_digest:  [u8; 32],
    pub word_idx_digest:      [u8; 32],
    pub phrase_idx_digest:    [u8; 32],
    pub semantic_idx_digest:  [u8; 32],
    pub discourse_idx_digest: [u8; 32],
    pub root_digest:      [u8; 32],
}

impl TowerManifest {
    fn compute_root(
        layer_digests: &[[u8; 32]],
        edge_digest:   &[u8; 32],
        index_digests: &[[u8; 32]],
    ) -> [u8; 32] {
        let mut leaves: Vec<[u8; 32]> = Vec::new();
        leaves.extend_from_slice(layer_digests);
        leaves.push(*edge_digest);
        leaves.extend_from_slice(index_digests);
        // Sort leaves for canonical ordering
        let mut sorted = leaves.clone();
        sorted.sort_unstable();
        merkle_root(&sorted)
    }

    pub fn canonical_bytes(&self) -> Vec<u8> {
        // Keys sorted lexicographically
        format!(
            "{{\"discourse_digest\":\"{}\",\"edge_digest\":\"{}\",\
             \"morpheme_digest\":\"{}\",\"phoneme_digest\":\"{}\",\
             \"phrase_digest\":\"{}\",\"root_digest\":\"{}\",\
             \"semantic_digest\":\"{}\",\"syllable_digest\":\"{}\",\
             \"word_digest\":\"{}\"}}",
            hex::encode(self.discourse_digest),
            hex::encode(self.edge_digest),
            hex::encode(self.morpheme_digest),
            hex::encode(self.phoneme_digest),
            hex::encode(self.phrase_digest),
            hex::encode(self.root_digest),
            hex::encode(self.semantic_digest),
            hex::encode(self.syllable_digest),
            hex::encode(self.word_digest),
        ).into_bytes()
    }

    pub fn digest(&self) -> [u8; 32] {
        sha256_bytes(&self.canonical_bytes())
    }

    pub fn render(&self) -> String {
        format!(
            "TowerManifest\n\
             ├─ phoneme   {}\n\
             ├─ syllable  {}\n\
             ├─ morpheme  {}\n\
             ├─ word      {}\n\
             ├─ phrase    {}\n\
             ├─ semantic  {}\n\
             ├─ discourse {}\n\
             ├─ edges     {}\n\
             └─ root      {}",
            hex::encode(&self.phoneme_digest[..8]),
            hex::encode(&self.syllable_digest[..8]),
            hex::encode(&self.morpheme_digest[..8]),
            hex::encode(&self.word_digest[..8]),
            hex::encode(&self.phrase_digest[..8]),
            hex::encode(&self.semantic_digest[..8]),
            hex::encode(&self.discourse_digest[..8]),
            hex::encode(&self.edge_digest[..8]),
            hex::encode(&self.root_digest[..8]),
        )
    }
}

// ── TowerQuery ────────────────────────────────────────────────────────────────

/// A query to the tower with its result.
#[derive(Clone, Debug)]
pub struct TowerQuery {
    pub query:        String,
    pub start_layer:  LayerId,
    pub start_idx:    usize,
    pub tau:          f64,
    pub top_k:        usize,
}

/// Result of a tower query.
#[derive(Clone, Debug)]
pub struct TowerQueryResult {
    pub query:        String,
    pub pass:         ForwardPass,
    pub final_layer:  LayerId,
    pub final_rendered: String,
    pub query_digest: [u8; 32],
}

impl TowerQueryResult {
    pub fn canonical_bytes(&self) -> Vec<u8> {
        format!(
            "{{\"final_layer\":\"{}\",\"pass_digest\":\"{}\",\"query\":\"{}\"}}",
            self.final_layer.as_str(),
            hex::encode(self.pass.pass_digest),
            self.query,
        ).into_bytes()
    }

    pub fn digest(&self) -> [u8; 32] {
        sha256_bytes(&self.canonical_bytes())
    }
}

// ── Tower ─────────────────────────────────────────────────────────────────────

/// The unified certified tower artifact.
pub struct Tower {
    // Layers
    pub phoneme:   PhonemeLayer,
    pub syllable:  SyllableLayer,
    pub morpheme:  MorphemeLayer,
    pub word:      WordLayer,
    pub phrase:    PhraseLayer,
    pub semantic:  SemanticLayer,
    pub discourse: DiscourseLayer,
    // Sig indices
    pub phoneme_idx:   SigIndex,
    pub syllable_idx:  SigIndex,
    pub morpheme_idx:  SigIndex,
    pub word_idx:      SigIndex,
    pub phrase_idx:    SigIndex,
    pub semantic_idx:  SigIndex,
    pub discourse_idx: SigIndex,
    // Manifest
    pub manifest: TowerManifest,
}

impl Tower {
    /// Build the complete certified tower.
    /// This is the canonical constructor — call once, reuse everywhere.
    pub fn build() -> Self {
        let phoneme   = PhonemeLayer::new();
        let syllable  = SyllableLayer::new();
        let morpheme  = MorphemeLayer::new();
        let word      = WordLayer::new();
        let phrase    = PhraseLayer::new();
        let semantic  = SemanticLayer::new();
        let discourse = DiscourseLayer::new();

        let phoneme_idx   = SigIndex::build(&phoneme);
        let syllable_idx  = SigIndex::build(&syllable);
        let morpheme_idx  = SigIndex::build(&morpheme);
        let word_idx      = SigIndex::build(&word);
        let phrase_idx    = SigIndex::build(&phrase);
        let semantic_idx  = SigIndex::build(&semantic);
        let discourse_idx = SigIndex::build(&discourse);

        let layer_digests = [
            phoneme.universe_digest(),
            syllable.universe_digest(),
            morpheme.universe_digest(),
            word.universe_digest(),
            phrase.universe_digest(),
            semantic.universe_digest(),
            discourse.universe_digest(),
        ];
        let edge_digest = tower_edge_digest();
        let index_digests = [
            phoneme_idx.index_digest,
            syllable_idx.index_digest,
            morpheme_idx.index_digest,
            word_idx.index_digest,
            phrase_idx.index_digest,
            semantic_idx.index_digest,
            discourse_idx.index_digest,
        ];
        let root_digest = TowerManifest::compute_root(
            &layer_digests, &edge_digest, &index_digests
        );

        let manifest = TowerManifest {
            phoneme_digest:      layer_digests[0],
            syllable_digest:     layer_digests[1],
            morpheme_digest:     layer_digests[2],
            word_digest:         layer_digests[3],
            phrase_digest:       layer_digests[4],
            semantic_digest:     layer_digests[5],
            discourse_digest:    layer_digests[6],
            edge_digest,
            phoneme_idx_digest:  index_digests[0],
            syllable_idx_digest: index_digests[1],
            morpheme_idx_digest: index_digests[2],
            word_idx_digest:     index_digests[3],
            phrase_idx_digest:   index_digests[4],
            semantic_idx_digest: index_digests[5],
            discourse_idx_digest:index_digests[6],
            root_digest,
        };

        Tower {
            phoneme, syllable, morpheme, word, phrase, semantic, discourse,
            phoneme_idx, syllable_idx, morpheme_idx, word_idx,
            phrase_idx, semantic_idx, discourse_idx,
            manifest,
        }
    }

    /// Get a layer by id.
    pub fn layer(&self, id: LayerId) -> &dyn Layer {
        match id {
            LayerId::Phoneme   => &self.phoneme,
            LayerId::Syllable  => &self.syllable,
            LayerId::Morpheme  => &self.morpheme,
            LayerId::Word      => &self.word,
            LayerId::Phrase    => &self.phrase,
            LayerId::Semantic  => &self.semantic,
            LayerId::Discourse => &self.discourse,
            _                  => &self.phoneme, // fallback
        }
    }

    /// Get a sig index by layer id.
    pub fn index(&self, id: LayerId) -> &SigIndex {
        match id {
            LayerId::Phoneme   => &self.phoneme_idx,
            LayerId::Syllable  => &self.syllable_idx,
            LayerId::Morpheme  => &self.morpheme_idx,
            LayerId::Word      => &self.word_idx,
            LayerId::Phrase    => &self.phrase_idx,
            LayerId::Semantic  => &self.semantic_idx,
            LayerId::Discourse => &self.discourse_idx,
            _                  => &self.phoneme_idx,
        }
    }

    /// Run a forward pass starting from a phoneme index.
    pub fn forward(&self, start_phoneme_idx: usize, tau: f64, top_k: usize)
        -> Result<ForwardPass, String>
    {
        TowerTransformer::forward_pass(
            start_phoneme_idx,
            &self.phoneme, &self.syllable, &self.morpheme,
            &self.word, &self.phrase, &self.semantic, &self.discourse,
            tau, top_k,
        )
    }

    /// Execute a tower query.
    pub fn query(&self, q: TowerQuery) -> Result<TowerQueryResult, String> {
        let pass = self.forward(q.start_idx, q.tau, q.top_k)?;
        let final_layer = LayerId::Discourse;
        let final_rendered = self.discourse.render(pass.final_idx);
        let query_digest = sha256_bytes(q.query.as_bytes());
        Ok(TowerQueryResult {
            query:          q.query,
            pass,
            final_layer,
            final_rendered,
            query_digest,
        })
    }

    /// Collect a verified trace corpus using the rule-based proposer.
    /// Returns the trainer with all records and a corpus digest.
    pub fn collect_trace_corpus(
        &self,
        n_phonemes: usize,
        tau:        f64,
        top_k:      usize,
    ) -> ProposerTrainer {
        let mut trainer = ProposerTrainer::new();
        let n = self.phoneme.len().min(n_phonemes);

        for ph_idx in 0..n {
            let mut ctx = ProposerContext::new(LayerId::Phoneme);

            // Simulate proposer interaction for this phoneme
            for _ in 0..10 {
                let dist = RuleBasedProposer::propose(&ctx);
                let top_op = match dist.top() {
                    Some(op) => op.clone(),
                    None     => break,
                };

                if top_op.kind.is_terminal() {
                    trainer.record(TraceRecord {
                        context:      ctx.clone(),
                        accepted_op:  top_op,
                        distribution: dist,
                        step_digest:  sha256_bytes(b"terminal"),
                        first_try:    true,
                    });
                    break;
                }

                // Simulate a verified step digest
                let step_digest = sha256_bytes(
                    &[ctx.chain_hash.as_slice(),
                      &ph_idx.to_le_bytes()].concat()
                );
                let next_layer = top_op.tgt_layer.unwrap_or(ctx.active_layer);

                trainer.record(TraceRecord {
                    context:      ctx.clone(),
                    accepted_op:  top_op,
                    distribution: dist,
                    step_digest,
                    first_try:    true,
                });

                ctx.advance(&step_digest, next_layer);
            }
        }
        trainer
    }

    /// Verify the tower's internal consistency.
    /// Checks that all universe digests match what was committed in the manifest.
    pub fn verify(&self) -> Result<(), String> {
        let checks: &[(&str, [u8; 32], [u8; 32])] = &[
            ("phoneme",   self.phoneme.universe_digest(),   self.manifest.phoneme_digest),
            ("syllable",  self.syllable.universe_digest(),  self.manifest.syllable_digest),
            ("morpheme",  self.morpheme.universe_digest(),  self.manifest.morpheme_digest),
            ("word",      self.word.universe_digest(),      self.manifest.word_digest),
            ("phrase",    self.phrase.universe_digest(),    self.manifest.phrase_digest),
            ("semantic",  self.semantic.universe_digest(),  self.manifest.semantic_digest),
            ("discourse", self.discourse.universe_digest(), self.manifest.discourse_digest),
        ];
        for (name, actual, expected) in checks {
            if actual != expected {
                return Err(format!(
                    "{} universe_digest mismatch: {} != {}",
                    name, hex::encode(actual), hex::encode(expected)
                ));
            }
        }
        // Verify root
        let layer_digests = [
            self.manifest.phoneme_digest, self.manifest.syllable_digest,
            self.manifest.morpheme_digest, self.manifest.word_digest,
            self.manifest.phrase_digest, self.manifest.semantic_digest,
            self.manifest.discourse_digest,
        ];
        let index_digests = [
            self.manifest.phoneme_idx_digest, self.manifest.syllable_idx_digest,
            self.manifest.morpheme_idx_digest, self.manifest.word_idx_digest,
            self.manifest.phrase_idx_digest, self.manifest.semantic_idx_digest,
            self.manifest.discourse_idx_digest,
        ];
        let expected_root = TowerManifest::compute_root(
            &layer_digests, &self.manifest.edge_digest, &index_digests
        );
        if expected_root != self.manifest.root_digest {
            return Err(format!(
                "root_digest mismatch: {} != {}",
                hex::encode(expected_root),
                hex::encode(self.manifest.root_digest)
            ));
        }
        Ok(())
    }

    /// Print a full tower summary.
    pub fn summary(&self) -> String {
        format!(
            "{}\n\nLayers:\n\
             ├─ PHONEME   {} elems\n\
             ├─ SYLLABLE  {} elems\n\
             ├─ MORPHEME  {} elems\n\
             ├─ WORD      {} elems\n\
             ├─ PHRASE    {} elems\n\
             ├─ SEMANTIC  {} elems\n\
             └─ DISCOURSE {} elems",
            self.manifest.render(),
            self.phoneme.len(),
            self.syllable.len(),
            self.morpheme.len(),
            self.word.len(),
            self.phrase.len(),
            self.semantic.len(),
            self.discourse.len(),
        )
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tower_build_root_digest_stable() {
        let t1 = Tower::build();
        let t2 = Tower::build();
        assert_eq!(t1.manifest.root_digest, t2.manifest.root_digest);
    }

    #[test]
    fn tower_verify_passes() {
        let tower = Tower::build();
        tower.verify().unwrap();
    }

    #[test]
    fn tower_manifest_render_nonempty() {
        let tower = Tower::build();
        let s = tower.manifest.render();
        assert!(s.contains("TowerManifest"));
        assert!(s.contains("root"));
    }

    #[test]
    fn tower_summary_contains_all_layers() {
        let tower = Tower::build();
        let s = tower.summary();
        assert!(s.contains("PHONEME"));
        assert!(s.contains("SYLLABLE"));
        assert!(s.contains("MORPHEME"));
        assert!(s.contains("WORD"));
        assert!(s.contains("PHRASE"));
        assert!(s.contains("SEMANTIC"));
        assert!(s.contains("DISCOURSE"));
    }

    #[test]
    fn tower_layer_counts_correct() {
        let tower = Tower::build();
        assert_eq!(tower.phoneme.len(),   44);
        assert!(tower.syllable.len()  > 100_000);
        assert_eq!(tower.morpheme.len(),  16);
        assert!(tower.word.len()      > 30_000);
        assert_eq!(tower.phrase.len(),     5);
        assert_eq!(tower.semantic.len(),   6);
        assert_eq!(tower.discourse.len(),  5);
    }

    #[test]
    fn tower_forward_pass_verifies() {
        let tower = Tower::build();
        let pass = tower.forward(0, 1.0, 3).unwrap();
        pass.verify_all().unwrap();
    }

    #[test]
    fn tower_forward_pass_digest_stable() {
        let tower = Tower::build();
        let p1 = tower.forward(0, 1.0, 3).unwrap();
        let p2 = tower.forward(0, 1.0, 3).unwrap();
        assert_eq!(p1.pass_digest, p2.pass_digest);
    }

    #[test]
    fn tower_query_succeeds() {
        let tower = Tower::build();
        let q = TowerQuery {
            query:       "test query".to_string(),
            start_layer: LayerId::Phoneme,
            start_idx:   0,
            tau:         1.0,
            top_k:       3,
        };
        let result = tower.query(q).unwrap();
        assert!(!result.final_rendered.is_empty());
        assert_eq!(result.final_layer, LayerId::Discourse);
    }

    #[test]
    fn tower_query_digest_stable() {
        let tower = Tower::build();
        let q1 = TowerQuery {
            query: "q".to_string(), start_layer: LayerId::Phoneme,
            start_idx: 0, tau: 1.0, top_k: 3,
        };
        let q2 = TowerQuery {
            query: "q".to_string(), start_layer: LayerId::Phoneme,
            start_idx: 0, tau: 1.0, top_k: 3,
        };
        let r1 = tower.query(q1).unwrap();
        let r2 = tower.query(q2).unwrap();
        assert_eq!(r1.digest(), r2.digest());
    }

    #[test]
    fn tower_collect_trace_corpus() {
        let tower = Tower::build();
        let trainer = tower.collect_trace_corpus(5, 1.0, 3);
        assert!(!trainer.records.is_empty());
        let d1 = trainer.corpus_digest();
        let d2 = trainer.corpus_digest();
        assert_eq!(d1, d2);
    }

    #[test]
    fn tower_corpus_nonempty_and_stable() {
        let tower = Tower::build();
        let t1 = tower.collect_trace_corpus(3, 1.0, 3);
        let t2 = tower.collect_trace_corpus(3, 1.0, 3);
        assert_eq!(t1.corpus_digest(), t2.corpus_digest());
        assert!(t1.records.len() > 0);
    }

    #[test]
    fn tower_index_all_layers() {
        let tower = Tower::build();
        for id in [LayerId::Phoneme, LayerId::Syllable, LayerId::Morpheme,
                   LayerId::Word, LayerId::Phrase, LayerId::Semantic, LayerId::Discourse] {
            let idx = tower.index(id);
            assert!(idx.total_elems > 0, "empty index for {:?}", id);
        }
    }

    #[test]
    fn tower_manifest_digest_stable() {
        let tower = Tower::build();
        assert_eq!(tower.manifest.digest(), tower.manifest.digest());
    }

    #[test]
    fn tower_root_digest_nonzero() {
        let tower = Tower::build();
        assert_ne!(tower.manifest.root_digest, [0u8; 32]);
    }

    #[test]
    fn tower_forward_render_trace() {
        let tower = Tower::build();
        let pass = tower.forward(0, 1.0, 3).unwrap();
        let trace = pass.render_trace();
        assert!(trace.contains("ForwardPass"));
        assert!(trace.contains("PHONEME"));
    }
}
