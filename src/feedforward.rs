//! Feed-forward sublayer as certified cross-layer projection — Stage 5.
//!
//! Standard FFN:
//!   FFN(x) = max(0, xW₁ + b₁)W₂ + b₂
//!   Facts stored in learned weight matrices W₁, W₂.
//!
//! This system:
//!   FFN(witness, src_layer, tgt_layer) = project_edge(src→tgt)(witness)
//!   Facts stored in certified tower universes.
//!   Traversal is deterministic, auditable, certified.
//!
//! A TowerFFN block:
//!   1. Takes a witness (elem index + layer)
//!   2. Projects it through the certified edge to the target layer
//!   3. Runs CertifiedAttention on the target layer using the projected sig
//!   4. Returns the attention result as the "activated" output
//!   5. Commits the entire operation to a step digest
//!
//! This replaces weight-based fact retrieval with universe-based fact retrieval.
//! The "activation function" is attention sharpness (tau), not ReLU.

use crate::digest::sha256_bytes;
use crate::layer::{Layer, LayerId};
use crate::edges::{
    EdgeId,
    PhonemeToSyllableEdge,
    SyllableToMorphemeEdge,
    MorphemeToWordEdge,
    WordToPhraseEdge,
    PhraseToSemanticEdge,
    SemanticToDiscourseEdge,
    ProjectionResult,
};
use crate::attention::{CertifiedAttention, AttentionResult};

// ── FFNStep ───────────────────────────────────────────────────────────────────

/// A single certified feed-forward step: one cross-layer projection + attention.
#[derive(Clone, Debug)]
pub struct FFNStep {
    pub edge:           EdgeId,
    pub src_elem_idx:   usize,
    pub src_rendered:   String,
    pub src_digest:     [u8; 32],
    pub projection:     ProjectionResult,
    pub attention:      AttentionResult,
    /// Digest committing this entire step.
    pub step_digest:    [u8; 32],
}

impl FFNStep {
    pub fn canonical_bytes(&self) -> Vec<u8> {
        format!(
            "{{\"attention_digest\":\"{}\",\"edge\":\"{}\",\"projection_digest\":\"{}\",\"src_digest\":\"{}\",\"src_elem_idx\":{}}}",
            hex::encode(self.attention.result_digest),
            self.edge.as_str(),
            hex::encode(self.projection.digest()),
            hex::encode(self.src_digest),
            self.src_elem_idx,
        ).into_bytes()
    }

    pub fn verify(&self) -> Result<(), String> {
        // 1. Verify attention
        self.attention.verify()?;
        // 2. Verify step digest
        let expected = sha256_bytes(&self.canonical_bytes());
        if expected != self.step_digest {
            return Err(format!(
                "step_digest mismatch: expected {} got {}",
                hex::encode(expected),
                hex::encode(self.step_digest),
            ));
        }
        Ok(())
    }
}

// ── TowerFFN ──────────────────────────────────────────────────────────────────

pub struct TowerFFN;

impl TowerFFN {
    /// Project a witness from src_layer to tgt_layer via the certified edge,
    /// then attend on tgt_layer using the projected signature.
    ///
    /// - `src_layer`: the source layer (left side of edge)
    /// - `tgt_layer`: the target layer (right side of edge)
    /// - `src_elem_idx`: index of the element in src_layer
    /// - `tau`: attention temperature
    /// - `top_k`: number of attention results to return
    pub fn forward(
        src_layer:    &dyn Layer,
        tgt_layer:    &dyn Layer,
        src_elem_idx: usize,
        tau:          f64,
        top_k:        usize,
    ) -> Result<FFNStep, String> {
        let edge_id = Self::edge_for(src_layer.id(), tgt_layer.id())?;
        let src_canonical = src_layer.canonical_bytes(src_elem_idx);
        let src_digest    = sha256_bytes(&src_canonical);
        let src_rendered  = src_layer.render(src_elem_idx);

        // Project: src elem → right-layer sig constraint
        let projection = Self::project(edge_id, src_layer, src_elem_idx, &src_canonical)?;

        // Attend on tgt_layer using projected sig
        let attention = CertifiedAttention::attend(
            tgt_layer,
            projection.right_sig,
            tau,
            top_k,
        );

        let mut step = FFNStep {
            edge: edge_id,
            src_elem_idx,
            src_rendered,
            src_digest,
            projection,
            attention,
            step_digest: [0u8; 32],
        };
        step.step_digest = sha256_bytes(&step.canonical_bytes());
        Ok(step)
    }

    /// Determine the EdgeId for a (src, tgt) layer pair.
    pub fn edge_for(src: LayerId, tgt: LayerId) -> Result<EdgeId, String> {
        match (src, tgt) {
            (LayerId::Phoneme,   LayerId::Syllable)  => Ok(EdgeId::PhonemeToSyllable),
            (LayerId::Syllable,  LayerId::Morpheme)  => Ok(EdgeId::SyllableToMorpheme),
            (LayerId::Morpheme,  LayerId::Word)      => Ok(EdgeId::MorphemeToWord),
            (LayerId::Word,      LayerId::Phrase)    => Ok(EdgeId::WordToPhrase),
            (LayerId::Phrase,    LayerId::Semantic)  => Ok(EdgeId::PhraseToSemantic),
            (LayerId::Semantic,  LayerId::Discourse) => Ok(EdgeId::SemanticToDiscourse),
            _ => Err(format!("no certified edge from {:?} to {:?}", src, tgt)),
        }
    }

    /// Dispatch projection through the correct edge type.
    fn project(
        edge_id:       EdgeId,
        src_layer:     &dyn Layer,
        src_elem_idx:  usize,
        src_canonical: &[u8],
    ) -> Result<ProjectionResult, String> {
        match edge_id {
            EdgeId::PhonemeToSyllable => {
                // phoneme id is the element index for phonemes
                let id = src_elem_idx as u8;
                Ok(PhonemeToSyllableEdge::project(id, src_canonical))
            }
            EdgeId::SyllableToMorpheme => {
                use crate::syllable::build_syllable_universe;
                let syllables = build_syllable_universe();
                let coda = syllables.get(src_elem_idx)
                    .map(|s| s.coda.clone())
                    .unwrap_or_default();
                Ok(SyllableToMorphemeEdge::project(&coda, src_canonical))
            }
            EdgeId::MorphemeToWord => {
                use crate::morpheme::build_morpheme_universe;
                let morphemes = build_morpheme_universe();
                let sig = morphemes.get(src_elem_idx)
                    .map(|m| crate::morpheme::sig16(m))
                    .unwrap_or(0);
                Ok(MorphemeToWordEdge::project(sig, src_canonical))
            }
            EdgeId::WordToPhrase => {
                use crate::word::build_word_universe;
                let words = build_word_universe();
                let word_sig = words.get(src_elem_idx)
                    .map(|w| w.sig)
                    .unwrap_or(0);
                Ok(WordToPhraseEdge::project(word_sig, src_canonical))
            }
            EdgeId::PhraseToSemantic => {
                let phrase_sig = src_layer.sig(src_elem_idx);
                Ok(PhraseToSemanticEdge::project(phrase_sig, src_canonical))
            }
            EdgeId::SemanticToDiscourse => {
                let sem_sig = src_layer.sig(src_elem_idx);
                Ok(SemanticToDiscourseEdge::project(sem_sig, src_canonical))
            }
        }
    }

    /// Run a full upward pass: PHONEME → SYLLABLE → MORPHEME → WORD → PHRASE → SEMANTIC → DISCOURSE
    /// Starting from a phoneme element index.
    /// Returns one FFNStep per edge (6 total).
    pub fn full_upward_pass(
        phoneme_idx:    usize,
        phoneme_layer:  &dyn Layer,
        syllable_layer: &dyn Layer,
        morpheme_layer: &dyn Layer,
        word_layer:     &dyn Layer,
        phrase_layer:   &dyn Layer,
        semantic_layer: &dyn Layer,
        discourse_layer:&dyn Layer,
        tau:            f64,
        top_k:          usize,
    ) -> Result<Vec<FFNStep>, String> {
        let mut steps = Vec::with_capacity(6);

        // Step 1: PHONEME → SYLLABLE
        let s1 = Self::forward(phoneme_layer, syllable_layer, phoneme_idx, tau, top_k)?;
        let syl_idx = s1.attention.top_idx;
        steps.push(s1);

        // Step 2: SYLLABLE → MORPHEME
        let s2 = Self::forward(syllable_layer, morpheme_layer, syl_idx, tau, top_k)?;
        let morph_idx = s2.attention.top_idx;
        steps.push(s2);

        // Step 3: MORPHEME → WORD
        let s3 = Self::forward(morpheme_layer, word_layer, morph_idx, tau, top_k)?;
        let word_idx = s3.attention.top_idx;
        steps.push(s3);

        // Step 4: WORD → PHRASE
        let s4 = Self::forward(word_layer, phrase_layer, word_idx, tau, top_k)?;
        let phrase_idx = s4.attention.top_idx;
        steps.push(s4);

        // Step 5: PHRASE → SEMANTIC
        let s5 = Self::forward(phrase_layer, semantic_layer, phrase_idx, tau, top_k)?;
        let sem_idx = s5.attention.top_idx;
        steps.push(s5);

        // Step 6: SEMANTIC → DISCOURSE
        let s6 = Self::forward(semantic_layer, discourse_layer, sem_idx, tau, top_k)?;
        steps.push(s6);

        Ok(steps)
    }

    /// Digest committing an entire sequence of FFN steps (a "block digest").
    pub fn block_digest(steps: &[FFNStep]) -> [u8; 32] {
        let mut combined = Vec::new();
        for s in steps {
            combined.extend_from_slice(&s.step_digest);
        }
        sha256_bytes(&combined)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::phoneme::PhonemeLayer;
    use crate::syllable::SyllableLayer;
    use crate::morpheme::MorphemeLayer;
    use crate::word::WordLayer;
    use crate::phrase::PhraseLayer;
    use crate::semantic::SemanticLayer;
    use crate::discourse::DiscourseLayer;
    use crate::layer::Layer;

    fn all_layers() -> (PhonemeLayer, SyllableLayer, MorphemeLayer, WordLayer,
                        PhraseLayer, SemanticLayer, DiscourseLayer) {
        (PhonemeLayer::new(), SyllableLayer::new(), MorphemeLayer::new(),
         WordLayer::new(), PhraseLayer::new(), SemanticLayer::new(), DiscourseLayer::new())
    }

    #[test]
    fn ffn_edge_for_all_valid() {
        assert!(TowerFFN::edge_for(LayerId::Phoneme,   LayerId::Syllable).is_ok());
        assert!(TowerFFN::edge_for(LayerId::Syllable,  LayerId::Morpheme).is_ok());
        assert!(TowerFFN::edge_for(LayerId::Morpheme,  LayerId::Word).is_ok());
        assert!(TowerFFN::edge_for(LayerId::Word,      LayerId::Phrase).is_ok());
        assert!(TowerFFN::edge_for(LayerId::Phrase,    LayerId::Semantic).is_ok());
        assert!(TowerFFN::edge_for(LayerId::Semantic,  LayerId::Discourse).is_ok());
    }

    #[test]
    fn ffn_edge_for_invalid() {
        assert!(TowerFFN::edge_for(LayerId::Discourse, LayerId::Phoneme).is_err());
        assert!(TowerFFN::edge_for(LayerId::Word,      LayerId::Semantic).is_err());
    }

    #[test]
    fn ffn_forward_phoneme_to_syllable() {
        let (ph, syl, ..) = all_layers();
        let step = TowerFFN::forward(&ph, &syl, 0, 1.0, 5).unwrap();
        assert_eq!(step.edge, EdgeId::PhonemeToSyllable);
        step.verify().unwrap();
    }

    #[test]
    fn ffn_forward_morpheme_to_word() {
        let (_, _, morph, word, _, _, _) = all_layers();
        let step = TowerFFN::forward(&morph, &word, 0, 1.0, 5).unwrap();
        assert_eq!(step.edge, EdgeId::MorphemeToWord);
        step.verify().unwrap();
    }

    #[test]
    fn ffn_forward_word_to_phrase() {
        let (_, _, _, word, phrase, _, _) = all_layers();
        let step = TowerFFN::forward(&word, &phrase, 0, 1.0, 3).unwrap();
        assert_eq!(step.edge, EdgeId::WordToPhrase);
        step.verify().unwrap();
    }

    #[test]
    fn ffn_forward_phrase_to_semantic() {
        let (_, _, _, _, phrase, sem, _) = all_layers();
        let step = TowerFFN::forward(&phrase, &sem, 0, 1.0, 3).unwrap();
        assert_eq!(step.edge, EdgeId::PhraseToSemantic);
        step.verify().unwrap();
    }

    #[test]
    fn ffn_forward_semantic_to_discourse() {
        let (_, _, _, _, _, sem, disc) = all_layers();
        let step = TowerFFN::forward(&sem, &disc, 0, 1.0, 3).unwrap();
        assert_eq!(step.edge, EdgeId::SemanticToDiscourse);
        step.verify().unwrap();
    }

    #[test]
    fn ffn_step_digest_stable() {
        let (ph, syl, ..) = all_layers();
        let s1 = TowerFFN::forward(&ph, &syl, 0, 1.0, 5).unwrap();
        let s2 = TowerFFN::forward(&ph, &syl, 0, 1.0, 5).unwrap();
        assert_eq!(s1.step_digest, s2.step_digest);
    }

    #[test]
    fn ffn_step_src_rendered_nonempty() {
        let (ph, syl, ..) = all_layers();
        let step = TowerFFN::forward(&ph, &syl, 0, 1.0, 5).unwrap();
        assert!(!step.src_rendered.is_empty());
    }

    #[test]
    fn ffn_full_upward_pass_length() {
        let (ph, syl, morph, word, phrase, sem, disc) = all_layers();
        let steps = TowerFFN::full_upward_pass(
            0, &ph, &syl, &morph, &word, &phrase, &sem, &disc, 1.0, 3
        ).unwrap();
        assert_eq!(steps.len(), 6);
    }

    #[test]
    fn ffn_full_upward_pass_all_verify() {
        let (ph, syl, morph, word, phrase, sem, disc) = all_layers();
        let steps = TowerFFN::full_upward_pass(
            0, &ph, &syl, &morph, &word, &phrase, &sem, &disc, 1.0, 3
        ).unwrap();
        for (i, step) in steps.iter().enumerate() {
            step.verify().unwrap_or_else(|e| panic!("step {} failed: {}", i, e));
        }
    }

    #[test]
    fn ffn_full_upward_pass_edges_in_order() {
        let (ph, syl, morph, word, phrase, sem, disc) = all_layers();
        let steps = TowerFFN::full_upward_pass(
            0, &ph, &syl, &morph, &word, &phrase, &sem, &disc, 1.0, 3
        ).unwrap();
        assert_eq!(steps[0].edge, EdgeId::PhonemeToSyllable);
        assert_eq!(steps[1].edge, EdgeId::SyllableToMorpheme);
        assert_eq!(steps[2].edge, EdgeId::MorphemeToWord);
        assert_eq!(steps[3].edge, EdgeId::WordToPhrase);
        assert_eq!(steps[4].edge, EdgeId::PhraseToSemantic);
        assert_eq!(steps[5].edge, EdgeId::SemanticToDiscourse);
    }

    #[test]
    fn ffn_block_digest_stable() {
        let (ph, syl, morph, word, phrase, sem, disc) = all_layers();
        let steps1 = TowerFFN::full_upward_pass(
            0, &ph, &syl, &morph, &word, &phrase, &sem, &disc, 1.0, 3
        ).unwrap();
        let steps2 = TowerFFN::full_upward_pass(
            0, &ph, &syl, &morph, &word, &phrase, &sem, &disc, 1.0, 3
        ).unwrap();
        assert_eq!(TowerFFN::block_digest(&steps1), TowerFFN::block_digest(&steps2));
    }

    #[test]
    fn ffn_block_digest_differs_by_phoneme() {
        let (ph, syl, morph, word, phrase, sem, disc) = all_layers();
        let steps0 = TowerFFN::full_upward_pass(
            0, &ph, &syl, &morph, &word, &phrase, &sem, &disc, 1.0, 3
        ).unwrap();
        let steps1 = TowerFFN::full_upward_pass(
            1, &ph, &syl, &morph, &word, &phrase, &sem, &disc, 1.0, 3
        ).unwrap();
        // Different starting phoneme → different block digest
        assert_ne!(TowerFFN::block_digest(&steps0), TowerFFN::block_digest(&steps1));
    }
}
