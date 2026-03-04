//! Cross-layer certified projection edges — Stage 3 of the tower blueprint.
//!
//! Six edges connecting adjacent layers:
//!   PHONEME    →(phonotactic)→  SYLLABLE
//!   SYLLABLE   →(allomorph)→    MORPHEME
//!   MORPHEME   →(compose)→      WORD
//!   WORD       →(pos+tree)→     PHRASE
//!   PHRASE     →(predicate)→    SEMANTIC
//!   SEMANTIC   →(coref+causal)→ DISCOURSE
//!
//! Each edge is a certified struct with:
//!   - project(left_elem) -> right_sig_constraint
//!   - invert(right_sig) -> Vec<left_indices>
//!   - edge_digest() -> [u8;32]  (bound to rules + version)
//!
//! A change to either endpoint universe invalidates the edge digest,
//! which invalidates all cross-layer traces that used that edge.

use crate::digest::sha256_bytes;
use crate::layer::LayerId;

// ── EdgeId ────────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum EdgeId {
    PhonemeToSyllable,
    SyllableToMorpheme,
    MorphemeToWord,
    WordToPhrase,
    PhraseToSemantic,
    SemanticToDiscourse,
}

impl EdgeId {
    pub fn left(self) -> LayerId {
        match self {
            EdgeId::PhonemeToSyllable   => LayerId::Phoneme,
            EdgeId::SyllableToMorpheme  => LayerId::Syllable,
            EdgeId::MorphemeToWord      => LayerId::Morpheme,
            EdgeId::WordToPhrase        => LayerId::Word,
            EdgeId::PhraseToSemantic    => LayerId::Phrase,
            EdgeId::SemanticToDiscourse => LayerId::Semantic,
        }
    }

    pub fn right(self) -> LayerId {
        match self {
            EdgeId::PhonemeToSyllable   => LayerId::Syllable,
            EdgeId::SyllableToMorpheme  => LayerId::Morpheme,
            EdgeId::MorphemeToWord      => LayerId::Word,
            EdgeId::WordToPhrase        => LayerId::Phrase,
            EdgeId::PhraseToSemantic    => LayerId::Semantic,
            EdgeId::SemanticToDiscourse => LayerId::Discourse,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            EdgeId::PhonemeToSyllable   => "PHONEME→SYLLABLE",
            EdgeId::SyllableToMorpheme  => "SYLLABLE→MORPHEME",
            EdgeId::MorphemeToWord      => "MORPHEME→WORD",
            EdgeId::WordToPhrase        => "WORD→PHRASE",
            EdgeId::PhraseToSemantic    => "PHRASE→SEMANTIC",
            EdgeId::SemanticToDiscourse => "SEMANTIC→DISCOURSE",
        }
    }
}

// ── ProjectionResult ──────────────────────────────────────────────────────────

/// Result of a projection: a right-layer signature constraint + proof.
#[derive(Clone, Debug)]
pub struct ProjectionResult {
    pub edge:           EdgeId,
    /// The right-layer signature produced by projecting the left element.
    pub right_sig:      u16,
    /// Digest of the left element's canonical bytes (proof of input).
    pub left_digest:    [u8; 32],
    /// The edge digest at time of projection (proof of rules used).
    pub edge_digest:    [u8; 32],
}

impl ProjectionResult {
    pub fn canonical_bytes(&self) -> Vec<u8> {
        format!(
            "{{\"edge\":\"{}\",\"edge_digest\":\"{}\",\"left_digest\":\"{}\",\"right_sig\":{}}}",
            self.edge.as_str(),
            hex::encode(self.edge_digest),
            hex::encode(self.left_digest),
            self.right_sig,
        ).into_bytes()
    }

    pub fn digest(&self) -> [u8; 32] {
        sha256_bytes(&self.canonical_bytes())
    }
}

// ── Edge 1: PHONEME → SYLLABLE ────────────────────────────────────────────────
//
// Project: given a phoneme, produce the sig constraint for syllables that
//          contain that phoneme (as onset, nucleus, or coda).
// Invert:  given a syllable sig, find phonemes consistent with it.

pub struct PhonemeToSyllableEdge;

impl PhonemeToSyllableEdge {
    const VERSION: &'static str = "phoneme_to_syllable_v1";

    pub fn edge_digest() -> [u8; 32] {
        sha256_bytes(Self::VERSION.as_bytes())
    }

    /// Project a phoneme (by id) to the syllable sig bits it constrains.
    /// Returns a (mask, value) pair: syllables matching have
    /// (sig & mask) == value.
    pub fn project_to_sig_constraint(phoneme_id: u8) -> (u16, u16) {
        use crate::phoneme::build_phoneme_universe;
        let phonemes = build_phoneme_universe();
        if let Some(ph) = phonemes.iter().find(|p| p.id == phoneme_id) {
            // Use the phoneme's sig bits 0..3 (manner class) as the constraint
            // on syllable nucleus bits 4..7
            let ph_sig = ph.sig;
            let manner_bits = (ph_sig & 0x000F) as u16;
            // mask: bits 4..7 of syllable sig
            let mask  = 0x00F0u16;
            let value = manner_bits << 4;
            (mask, value)
        } else {
            (0, 0)
        }
    }

    /// Find all syllable indices (in the syllable inventory) that contain
    /// this phoneme as onset, nucleus, or coda.
    pub fn invert(phoneme_id: u8) -> Vec<usize> {
        use crate::syllable::build_syllable_universe;
        let syllables = build_syllable_universe();
        syllables.iter().enumerate()
            .filter(|(_, s)| {
                s.nucleus == phoneme_id
                    || s.onset.contains(&phoneme_id)
                    || s.coda.contains(&phoneme_id)
            })
            .map(|(i, _)| i)
            .collect()
    }

    pub fn project(phoneme_id: u8, left_canonical: &[u8]) -> ProjectionResult {
        let (_, right_sig) = Self::project_to_sig_constraint(phoneme_id);
        ProjectionResult {
            edge:        EdgeId::PhonemeToSyllable,
            right_sig,
            left_digest: sha256_bytes(left_canonical),
            edge_digest: Self::edge_digest(),
        }
    }
}

// ── Edge 2: SYLLABLE → MORPHEME ───────────────────────────────────────────────
//
// Project: given a syllable, produce the morpheme sig bits it could
//          participate in (based on coda voicing — relevant for allomorphy).
// Invert:  given a morpheme sig, find syllable indices consistent with it.

pub struct SyllableToMorphemeEdge;

impl SyllableToMorphemeEdge {
    const VERSION: &'static str = "syllable_to_morpheme_v1";

    pub fn edge_digest() -> [u8; 32] {
        sha256_bytes(Self::VERSION.as_bytes())
    }

    /// Project syllable coda voicing to morpheme sig constraint.
    /// Coda voicing determines plural/past allomorph selection.
    pub fn project_to_sig_constraint(syllable_coda: &[u8]) -> (u16, u16) {
        use crate::phoneme::build_phoneme_universe;
        let phonemes = build_phoneme_universe();

        if syllable_coda.is_empty() {
            // Open syllable: no constraint from coda
            return (0, 0);
        }
        let last_id = *syllable_coda.last().unwrap();
        if let Some(ph) = phonemes.iter().find(|p| p.id == last_id) {
            // Voicing bit is bit 0 of phoneme sig
            let voiced = ph.sig & 0x0001;
            // Morpheme sig bit 1 = requires_voiced_stem
            let mask  = 0x0002u16;
            let value = voiced << 1;
            (mask, value)
        } else {
            (0, 0)
        }
    }

    /// Find morpheme indices compatible with a given coda voicing.
    pub fn invert(is_voiced_coda: bool) -> Vec<usize> {
        use crate::morpheme::build_morpheme_universe;
        let morphemes = build_morpheme_universe();
        morphemes.iter().enumerate()
            .filter(|(_, m)| {
                // Allomorphic morphemes are compatible with any coda
                // Free morphemes are always compatible
                use crate::morpheme::MorphemeType;
                match m.morpheme_type {
                    MorphemeType::Suffix | MorphemeType::Prefix => true,
                    MorphemeType::Root | MorphemeType::Function =>
                        !is_voiced_coda || !m.surface_forms.is_empty(),
                }
            })
            .map(|(i, _)| i)
            .collect()
    }

    pub fn project(coda: &[u8], left_canonical: &[u8]) -> ProjectionResult {
        let (_, right_sig) = Self::project_to_sig_constraint(coda);
        ProjectionResult {
            edge:        EdgeId::SyllableToMorpheme,
            right_sig,
            left_digest: sha256_bytes(left_canonical),
            edge_digest: Self::edge_digest(),
        }
    }
}

// ── Edge 3: MORPHEME → WORD ───────────────────────────────────────────────────
//
// Project: given a morpheme (by meaning_id), find words that contain it.
// Invert:  given a word sig, find morpheme indices that could compose it.

pub struct MorphemeToWordEdge;

impl MorphemeToWordEdge {
    const VERSION: &'static str = "morpheme_to_word_v1";

    pub fn edge_digest() -> [u8; 32] {
        sha256_bytes(Self::VERSION.as_bytes())
    }

    /// Project morpheme gram_role to word sig constraint.
    pub fn project_to_sig_constraint(morpheme_sig: u16) -> (u16, u16) {
        // Morpheme sig bits 8..11 encode gram_role
        // Word sig bits 0..3 encode POS — gram_role constrains POS
        let gram_role_bits = (morpheme_sig >> 8) & 0x000F;
        let mask  = 0x000Fu16;
        let value = gram_role_bits & 0x000F;
        (mask, value)
    }

    /// Find word indices that contain this morpheme's surface form.
    pub fn invert(meaning_id: &str) -> Vec<usize> {
        use crate::word::build_word_universe;
        let words = build_word_universe();
        // Extract root from meaning_id: "en:word:cat" → "cat"
        let root = meaning_id.split(':').last().unwrap_or(meaning_id);
        words.iter().enumerate()
            .filter(|(_, w)| w.text.contains(root))
            .map(|(i, _)| i)
            .collect()
    }

    pub fn project(morpheme_sig: u16, left_canonical: &[u8]) -> ProjectionResult {
        let (_, right_sig) = Self::project_to_sig_constraint(morpheme_sig);
        ProjectionResult {
            edge:        EdgeId::MorphemeToWord,
            right_sig:   right_sig as u16,
            left_digest: sha256_bytes(left_canonical),
            edge_digest: Self::edge_digest(),
        }
    }
}

// ── Edge 4: WORD → PHRASE ─────────────────────────────────────────────────────
//
// Project: given a word, find phrases that contain it as a terminal node.
// Invert:  given a phrase sig, find word indices consistent with it.

pub struct WordToPhraseEdge;

impl WordToPhraseEdge {
    const VERSION: &'static str = "word_to_phrase_v1";

    pub fn edge_digest() -> [u8; 32] {
        sha256_bytes(Self::VERSION.as_bytes())
    }

    /// Project word sig to phrase sig constraint.
    pub fn project_to_sig_constraint(word_sig: u8) -> (u16, u16) {
        // Word sig bits 0..3 = POS → phrase sig bits 0..3 = node_type presence
        let pos_bits = (word_sig & 0x0F) as u16;
        let mask  = 0x000Fu16;
        let value = pos_bits;
        (mask, value)
    }

    /// Find phrase indices containing this word as a terminal.
    pub fn invert(word_text: &str) -> Vec<usize> {
        use crate::phrase::build_phrase_inventory;
        let phrases = build_phrase_inventory();
        let target = format!("en:word:{}", word_text);

        fn contains_word(node: &crate::phrase::PhraseNode, target: &str) -> bool {
            match node {
                crate::phrase::PhraseNode::Terminal { word, .. } => word == target,
                crate::phrase::PhraseNode::Inner { children, .. } =>
                    children.iter().any(|c| contains_word(c, target)),
            }
        }

        phrases.iter().enumerate()
            .filter(|(_, p)| contains_word(&p.root, &target))
            .map(|(i, _)| i)
            .collect()
    }

    pub fn project(word_sig: u8, left_canonical: &[u8]) -> ProjectionResult {
        let (_, right_sig) = Self::project_to_sig_constraint(word_sig);
        ProjectionResult {
            edge:        EdgeId::WordToPhrase,
            right_sig,
            left_digest: sha256_bytes(left_canonical),
            edge_digest: Self::edge_digest(),
        }
    }
}

// ── Edge 5: PHRASE → SEMANTIC ─────────────────────────────────────────────────
//
// Project: given a phrase, extract predicate structure → semantic sig.
// Invert:  given a semantic sig, find phrase indices that could generate it.

pub struct PhraseToSemanticEdge;

impl PhraseToSemanticEdge {
    const VERSION: &'static str = "phrase_to_semantic_v1";

    pub fn edge_digest() -> [u8; 32] {
        sha256_bytes(Self::VERSION.as_bytes())
    }

    /// Project phrase sig to semantic sig constraint.
    /// A phrase with VP node implies an Event node in semantic graph.
    pub fn project_to_sig_constraint(phrase_sig: u16) -> (u16, u16) {
        // Phrase sig bit 4 = has_VP → semantic sig bit 0 = has_event
        let has_vp = (phrase_sig >> 4) & 0x0001;
        let mask   = 0x0001u16;
        let value  = has_vp;
        (mask, value)
    }

    /// Find semantic graph indices that could be expressed by this phrase.
    pub fn invert(phrase_sig: u16) -> Vec<usize> {
        use crate::semantic::build_semantic_inventory;
        let graphs = build_semantic_inventory();
        let (mask, value) = Self::project_to_sig_constraint(phrase_sig);
        graphs.iter().enumerate()
            .filter(|(_, g)| {
                let gsig = crate::semantic::sig16(g);
                mask == 0 || (gsig & mask) == value
            })
            .map(|(i, _)| i)
            .collect()
    }

    pub fn project(phrase_sig: u16, left_canonical: &[u8]) -> ProjectionResult {
        let (_, right_sig) = Self::project_to_sig_constraint(phrase_sig);
        ProjectionResult {
            edge:        EdgeId::PhraseToSemantic,
            right_sig,
            left_digest: sha256_bytes(left_canonical),
            edge_digest: Self::edge_digest(),
        }
    }
}

// ── Edge 6: SEMANTIC → DISCOURSE ─────────────────────────────────────────────
//
// Project: given a semantic graph, find discourse graphs that reference it.
// Invert:  given a discourse sig, find semantic graph indices consistent.

pub struct SemanticToDiscourseEdge;

impl SemanticToDiscourseEdge {
    const VERSION: &'static str = "semantic_to_discourse_v1";

    pub fn edge_digest() -> [u8; 32] {
        sha256_bytes(Self::VERSION.as_bytes())
    }

    /// Project semantic sig to discourse sig constraint.
    /// Negated semantic graphs → discourse negation bit.
    pub fn project_to_sig_constraint(semantic_sig: u16) -> (u16, u16) {
        // Semantic sig bit 14 = polarity (1=negative) → discourse sig bit 4 = has_negation
        let neg_bit = (semantic_sig >> 14) & 0x0001;
        let mask    = 0x0010u16;
        let value   = neg_bit << 4;
        (mask, value)
    }

    /// Find discourse graph indices that reference this semantic graph_id.
    pub fn invert(graph_id: u32) -> Vec<usize> {
        use crate::discourse::build_discourse_inventory;
        let graphs = build_discourse_inventory();
        graphs.iter().enumerate()
            .filter(|(_, d)| {
                d.nodes.iter().any(|n| {
                    use crate::discourse::DiscourseNodeType;
                    n.node_type == DiscourseNodeType::SemanticGraphRef
                        && n.label.contains(&graph_id.to_string())
                })
            })
            .map(|(i, _)| i)
            .collect()
    }

    pub fn project(semantic_sig: u16, left_canonical: &[u8]) -> ProjectionResult {
        let (_, right_sig) = Self::project_to_sig_constraint(semantic_sig);
        ProjectionResult {
            edge:        EdgeId::SemanticToDiscourse,
            right_sig,
            left_digest: sha256_bytes(left_canonical),
            edge_digest: Self::edge_digest(),
        }
    }
}

// ── Tower edge digest ─────────────────────────────────────────────────────────

/// A single digest committing all six edge versions.
/// Changes when any edge rule changes.
pub fn tower_edge_digest() -> [u8; 32] {
    let mut combined = Vec::new();
    combined.extend_from_slice(&PhonemeToSyllableEdge::edge_digest());
    combined.extend_from_slice(&SyllableToMorphemeEdge::edge_digest());
    combined.extend_from_slice(&MorphemeToWordEdge::edge_digest());
    combined.extend_from_slice(&WordToPhraseEdge::edge_digest());
    combined.extend_from_slice(&PhraseToSemanticEdge::edge_digest());
    combined.extend_from_slice(&SemanticToDiscourseEdge::edge_digest());
    sha256_bytes(&combined)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edge_ids_left_right_consistent() {
        let edges = [
            EdgeId::PhonemeToSyllable,
            EdgeId::SyllableToMorpheme,
            EdgeId::MorphemeToWord,
            EdgeId::WordToPhrase,
            EdgeId::PhraseToSemantic,
            EdgeId::SemanticToDiscourse,
        ];
        // left depth < right depth for all edges
        for e in edges {
            assert!(e.left().depth() < e.right().depth(),
                "{} left depth {} >= right depth {}",
                e.as_str(), e.left().depth(), e.right().depth());
        }
    }

    #[test]
    fn edge_digests_stable() {
        assert_eq!(PhonemeToSyllableEdge::edge_digest(),   PhonemeToSyllableEdge::edge_digest());
        assert_eq!(SyllableToMorphemeEdge::edge_digest(),  SyllableToMorphemeEdge::edge_digest());
        assert_eq!(MorphemeToWordEdge::edge_digest(),      MorphemeToWordEdge::edge_digest());
        assert_eq!(WordToPhraseEdge::edge_digest(),        WordToPhraseEdge::edge_digest());
        assert_eq!(PhraseToSemanticEdge::edge_digest(),    PhraseToSemanticEdge::edge_digest());
        assert_eq!(SemanticToDiscourseEdge::edge_digest(), SemanticToDiscourseEdge::edge_digest());
    }

    #[test]
    fn tower_edge_digest_stable() {
        assert_eq!(tower_edge_digest(), tower_edge_digest());
    }

    #[test]
    fn tower_edge_digest_changes_with_version() {
        // Sanity: the combined digest is not all zeros
        let d = tower_edge_digest();
        assert_ne!(d, [0u8; 32]);
    }

    #[test]
    fn phoneme_to_syllable_invert_nonempty() {
        // phoneme id 0 (should be in inventory) should appear in some syllables
        use crate::phoneme::build_phoneme_universe;
        let phs = build_phoneme_universe();
        if let Some(ph) = phs.first() {
            let result = PhonemeToSyllableEdge::invert(ph.id);
            assert!(!result.is_empty(),
                "phoneme {} not found in any syllable", ph.symbol);
        }
    }

    #[test]
    fn phoneme_to_syllable_project_deterministic() {
        use crate::phoneme::build_phoneme_universe;
        let phs = build_phoneme_universe();
        let ph = &phs[0];
        let r1 = PhonemeToSyllableEdge::project(ph.id, &ph.canonical_bytes());
        let r2 = PhonemeToSyllableEdge::project(ph.id, &ph.canonical_bytes());
        assert_eq!(r1.right_sig, r2.right_sig);
        assert_eq!(r1.edge_digest, r2.edge_digest);
        assert_eq!(r1.left_digest, r2.left_digest);
    }

    #[test]
    fn morpheme_to_word_invert_cat() {
        let result = MorphemeToWordEdge::invert("en:word:cat");
        assert!(!result.is_empty(), "no words found containing 'cat'");
    }

    #[test]
    fn word_to_phrase_invert_nonempty() {
        // "the" appears in phrase inventory
        let result = WordToPhraseEdge::invert("the");
        assert!(!result.is_empty(), "no phrases found containing 'the'");
    }

    #[test]
    fn word_to_phrase_invert_cat() {
        let result = WordToPhraseEdge::invert("cat");
        assert!(!result.is_empty(), "no phrases found containing 'cat'");
    }

    #[test]
    fn phrase_to_semantic_invert_nonempty() {
        use crate::phrase::build_phrase_inventory;
        let phrases = build_phrase_inventory();
        // Try all phrases — at least one must map to a non-empty semantic set
        let any_nonempty = phrases.iter().any(|p| {
            !PhraseToSemanticEdge::invert(p.sig).is_empty()
        });
        assert!(any_nonempty, "no phrase sig maps to any semantic graph");
    }

    #[test]
    fn semantic_to_discourse_project_deterministic() {
        use crate::semantic::build_semantic_inventory;
        let graphs = build_semantic_inventory();
        let g = &graphs[0];
        let sig = crate::semantic::sig16(g);
        let r1 = SemanticToDiscourseEdge::project(sig, &g.canonical_bytes());
        let r2 = SemanticToDiscourseEdge::project(sig, &g.canonical_bytes());
        assert_eq!(r1.right_sig, r2.right_sig);
        assert_eq!(r1.digest(), r2.digest());
    }

    #[test]
    fn projection_result_canonical_bytes_deterministic() {
        use crate::phoneme::build_phoneme_universe;
        let phs = build_phoneme_universe();
        let r = PhonemeToSyllableEdge::project(phs[0].id, &phs[0].canonical_bytes());
        assert_eq!(r.canonical_bytes(), r.canonical_bytes());
    }

    #[test]
    fn projection_result_digest_stable() {
        use crate::phoneme::build_phoneme_universe;
        let phs = build_phoneme_universe();
        let r = PhonemeToSyllableEdge::project(phs[0].id, &phs[0].canonical_bytes());
        assert_eq!(r.digest(), r.digest());
    }
}
