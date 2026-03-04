//! Layer trait — unified interface over all 7 certified linguistic universes.
//!
//! Every universe implements `Layer`. The executor and verifier dispatch
//! through `AnyLayer` (a boxed trait object) instead of a chain of
//! `is_boolfun`/`is_word`/... booleans.
//!
//! Core contract:
//!   - `canonical_bytes` is the identity function: same structure → same bytes
//!   - `digest` = SHA-256(canonical_bytes)
//!   - `project` maps an element to its structural signature (lossy, for indexing)
//!   - `sig_distance` is the Hamming distance in signature space (integer, exact)
//!   - `render` produces a stable, human-readable, round-trip-safe string
//!   - `universe_digest` commits the full inventory + rules to a single [u8;32]

use crate::digest::sha256_bytes;

// ── LayerId ───────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, PartialOrd, Ord)]
pub enum LayerId {
    Phoneme,
    Syllable,
    Morpheme,
    Word,
    Phrase,
    Semantic,
    Discourse,
    // Non-linguistic universes (kept for completeness)
    BoolFun,
    QE,
}

impl LayerId {
    pub fn as_str(self) -> &'static str {
        match self {
            LayerId::Phoneme   => "PHONEME",
            LayerId::Syllable  => "SYLLABLE",
            LayerId::Morpheme  => "MORPHEME",
            LayerId::Word      => "WORD",
            LayerId::Phrase    => "PHRASE",
            LayerId::Semantic  => "SEMANTIC",
            LayerId::Discourse => "DISCOURSE",
            LayerId::BoolFun   => "BOOLFUN",
            LayerId::QE        => "QE",
        }
    }

    pub fn depth(self) -> u8 {
        match self {
            LayerId::Phoneme   => 1,
            LayerId::Syllable  => 2,
            LayerId::Morpheme  => 3,
            LayerId::Word      => 4,
            LayerId::Phrase    => 5,
            LayerId::Semantic  => 6,
            LayerId::Discourse => 7,
            LayerId::BoolFun   => 0,
            LayerId::QE        => 0,
        }
    }

    pub fn from_str(s: &str) -> Option<LayerId> {
        match s.to_ascii_uppercase().trim_end_matches(|c: char| c == ';' || c == ',') {
            "PHONEME" | "PH" | "PHONE"          => Some(LayerId::Phoneme),
            "SYLLABLE" | "SYL"                   => Some(LayerId::Syllable),
            "MORPHEME" | "MORPH"                 => Some(LayerId::Morpheme),
            "WORD" | "EN"                        => Some(LayerId::Word),
            "PHRASE" | "PHR"                     => Some(LayerId::Phrase),
            "SEMANTIC" | "SEM" | "SEMGRAPH"
            | "GRAPH"                            => Some(LayerId::Semantic),
            "DISCOURSE" | "DIS" | "DISC"         => Some(LayerId::Discourse),
            "BOOLFUN" | "BF" | "BOOL"            => Some(LayerId::BoolFun),
            "QE"                                 => Some(LayerId::QE),
            _                                    => None,
        }
    }
}

// ── TowerPosition ─────────────────────────────────────────────────────────────

/// The certified position of an element in the tower.
/// Replaces sinusoidal position embeddings with structural coordinates.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TowerPosition {
    pub layer:        LayerId,
    pub sequence_pos: u32,   // linear position in discourse; 0 for non-sequential layers
    pub tree_depth:   u8,    // node depth in parse tree; 0 for flat layers
    pub sig:          u16,   // structural position in sig space
}

impl TowerPosition {
    pub fn flat(layer: LayerId, sig: u16) -> Self {
        TowerPosition { layer, sequence_pos: 0, tree_depth: 0, sig }
    }

    pub fn canonical_bytes(&self) -> Vec<u8> {
        // Keys sorted: layer, sequence_pos, sig, tree_depth
        format!(
            "{{\"layer\":\"{}\",\"sequence_pos\":{},\"sig\":{},\"tree_depth\":{}}}",
            self.layer.as_str(), self.sequence_pos, self.sig, self.tree_depth
        ).into_bytes()
    }

    pub fn digest(&self) -> [u8; 32] {
        sha256_bytes(&self.canonical_bytes())
    }
}

// ── Layer witness ─────────────────────────────────────────────────────────────

/// A witness is a certified element from a layer, with its position.
#[derive(Clone, Debug)]
pub struct LayerWitness {
    pub layer:    LayerId,
    pub rendered: String,       // canonical human-readable form
    pub digest:   [u8; 32],     // SHA-256(canonical_bytes(elem))
    pub position: TowerPosition,
}

impl LayerWitness {
    pub fn canonical_bytes(&self) -> Vec<u8> {
        format!(
            "{{\"digest\":\"{}\",\"layer\":\"{}\",\"rendered\":\"{}\"}}",
            hex::encode(self.digest),
            self.layer.as_str(),
            self.rendered,
        ).into_bytes()
    }
}

// ── Layer trait ───────────────────────────────────────────────────────────────

/// The unified interface every certified universe must implement.
pub trait Layer: Send + Sync {
    /// Human-readable name for this layer.
    fn id(&self) -> LayerId;

    /// Number of elements in the certified inventory.
    fn len(&self) -> usize;

    fn is_empty(&self) -> bool { self.len() == 0 }

    /// Canonical byte encoding of element at index `i`.
    /// Deterministic: same structure → same bytes on any platform.
    fn canonical_bytes(&self, i: usize) -> Vec<u8>;

    /// SHA-256 of canonical_bytes(i).
    fn digest(&self, i: usize) -> [u8; 32] {
        sha256_bytes(&self.canonical_bytes(i))
    }

    /// 16-bit structural signature (lossy projection, for indexing).
    fn sig(&self, i: usize) -> u16;

    /// Hamming distance between two signatures.
    fn sig_distance(&self, a: u16, b: u16) -> u32 {
        (a ^ b).count_ones()
    }

    /// Canonical human-readable rendering of element i.
    /// Must be stable and round-trip-safe (render → parse → canonical_bytes → digest = identity).
    fn render(&self, i: usize) -> String;

    /// Merkle root over all element digests, bound to layer rules.
    fn universe_digest(&self) -> [u8; 32];

    /// Find the index of the element nearest to `target_sig` by Hamming distance.
    /// Ties broken by canonical index order (deterministic).
    fn nearest(&self, target_sig: u16) -> Option<usize> {
        (0..self.len())
            .min_by_key(|&i| self.sig_distance(self.sig(i), target_sig))
    }

    /// Find the top-k elements nearest to `target_sig`.
    fn top_k(&self, target_sig: u16, k: usize) -> Vec<usize> {
        let mut scored: Vec<(u32, usize)> = (0..self.len())
            .map(|i| (self.sig_distance(self.sig(i), target_sig), i))
            .collect();
        scored.sort_by(|(da, ia), (db, ib)| da.cmp(db).then(ia.cmp(ib)));
        scored.into_iter().take(k).map(|(_, i)| i).collect()
    }

    /// Sample up to `n` elements from the inventory (canonical order).
    fn sample(&self, n: usize) -> Vec<usize> {
        (0..self.len().min(n)).collect()
    }

    /// Build a LayerWitness for element at index `i`.
    fn witness(&self, i: usize) -> LayerWitness {
        let sig = self.sig(i);
        LayerWitness {
            layer:    self.id(),
            rendered: self.render(i),
            digest:   self.digest(i),
            position: TowerPosition::flat(self.id(), sig),
        }
    }
}

// ── TowerContext ──────────────────────────────────────────────────────────────

/// Runtime state for a semtrace execution.
/// Replaces the chain of `is_boolfun`/`is_word`/... booleans in exec.rs.
pub struct TowerContext {
    pub active_layer: LayerId,
    pub witness_idx:  Option<usize>,
    pub set_size:     usize,
    pub set_digest:   [u8; 32],
    pub chain_hash:   [u8; 32],
}

impl TowerContext {
    pub fn new() -> Self {
        TowerContext {
            active_layer: LayerId::QE,
            witness_idx:  None,
            set_size:     0,
            set_digest:   sha256_bytes(b""),
            chain_hash:   sha256_bytes(b""),
        }
    }

    pub fn witness_rendered(&self, layer: &dyn Layer) -> Option<String> {
        self.witness_idx.map(|i| layer.render(i))
    }
}

impl Default for TowerContext {
    fn default() -> Self { Self::new() }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layer_id_roundtrip() {
        let ids = [
            LayerId::Phoneme, LayerId::Syllable, LayerId::Morpheme,
            LayerId::Word, LayerId::Phrase, LayerId::Semantic, LayerId::Discourse,
        ];
        for id in ids {
            let s = id.as_str();
            assert_eq!(LayerId::from_str(s), Some(id), "roundtrip failed for {:?}", id);
        }
    }

    #[test]
    fn layer_id_aliases() {
        assert_eq!(LayerId::from_str("SYL"), Some(LayerId::Syllable));
        assert_eq!(LayerId::from_str("MORPH"), Some(LayerId::Morpheme));
        assert_eq!(LayerId::from_str("SEM"), Some(LayerId::Semantic));
        assert_eq!(LayerId::from_str("DISC"), Some(LayerId::Discourse));
        assert_eq!(LayerId::from_str("unknown"), None);
    }

    #[test]
    fn layer_id_depth_ordered() {
        assert!(LayerId::Phoneme.depth() < LayerId::Syllable.depth());
        assert!(LayerId::Syllable.depth() < LayerId::Morpheme.depth());
        assert!(LayerId::Morpheme.depth() < LayerId::Word.depth());
        assert!(LayerId::Word.depth() < LayerId::Phrase.depth());
        assert!(LayerId::Phrase.depth() < LayerId::Semantic.depth());
        assert!(LayerId::Semantic.depth() < LayerId::Discourse.depth());
    }

    #[test]
    fn tower_position_canonical_bytes_deterministic() {
        let p = TowerPosition::flat(LayerId::Semantic, 0b1010_1010_1010_1010);
        let b1 = p.canonical_bytes();
        let b2 = p.canonical_bytes();
        assert_eq!(b1, b2);
    }

    #[test]
    fn tower_position_digest_deterministic() {
        let p = TowerPosition::flat(LayerId::Word, 42);
        assert_eq!(p.digest(), p.digest());
    }

    #[test]
    fn sig_distance_zero_self() {
        // Any Layer impl: distance(sig, sig) = 0
        // Test the default impl directly
        struct Stub;
        impl Layer for Stub {
            fn id(&self) -> LayerId { LayerId::Word }
            fn len(&self) -> usize { 0 }
            fn canonical_bytes(&self, _: usize) -> Vec<u8> { vec![] }
            fn sig(&self, _: usize) -> u16 { 0 }
            fn render(&self, _: usize) -> String { String::new() }
            fn universe_digest(&self) -> [u8; 32] { [0u8; 32] }
        }
        let s = Stub;
        assert_eq!(s.sig_distance(0xABCD, 0xABCD), 0);
    }

    #[test]
    fn sig_distance_max() {
        struct Stub;
        impl Layer for Stub {
            fn id(&self) -> LayerId { LayerId::Word }
            fn len(&self) -> usize { 0 }
            fn canonical_bytes(&self, _: usize) -> Vec<u8> { vec![] }
            fn sig(&self, _: usize) -> u16 { 0 }
            fn render(&self, _: usize) -> String { String::new() }
            fn universe_digest(&self) -> [u8; 32] { [0u8; 32] }
        }
        let s = Stub;
        // 0x0000 vs 0xFFFF = 16 bits flipped
        assert_eq!(s.sig_distance(0x0000, 0xFFFF), 16);
    }

    #[test]
    fn tower_context_default() {
        let ctx = TowerContext::new();
        assert_eq!(ctx.active_layer, LayerId::QE);
        assert!(ctx.witness_idx.is_none());
    }
}
