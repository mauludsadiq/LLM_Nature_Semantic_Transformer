//! Signature Inverted Index — Stage 1 of the tower scaling blueprint.
//!
//! For each universe, maps:
//!   Sig (u16) -> (posting_list_digest: [u8;32], elem_indices: Vec<usize>)
//!
//! This turns WITNESS_NEAREST from O(n) full scan to O(matching sigs + output).
//!
//! The index is itself a certified artifact:
//!   index_digest = MerkleRoot(sorted[(sig_be_bytes || posting_list_digest)])
//!
//! Verifier can check: "result is exactly the posting list for this sig
//! constraint, and that posting list has this digest." Proof is O(log n).

use std::collections::BTreeMap;
use crate::digest::{sha256_bytes, merkle_root};
use crate::layer::Layer;

// ── PostingList ───────────────────────────────────────────────────────────────

/// A certified posting list for a single signature value.
#[derive(Clone, Debug)]
pub struct PostingList {
    /// The signature this list covers.
    pub sig: u16,
    /// Sorted element indices in the inventory.
    pub indices: Vec<usize>,
    /// SHA-256(sorted canonical bytes of all elements in this list).
    pub digest: [u8; 32],
}

impl PostingList {
    fn build(sig: u16, mut indices: Vec<usize>, layer: &dyn Layer) -> Self {
        indices.sort_unstable();
        let mut leaves: Vec<[u8; 32]> = indices.iter()
            .map(|&i| sha256_bytes(&layer.canonical_bytes(i)))
            .collect();
        leaves.sort_unstable();
        let digest = merkle_root(&leaves);
        PostingList { sig, indices, digest }
    }

    pub fn canonical_bytes(&self) -> Vec<u8> {
        // sig (2 bytes BE) || posting_digest (32 bytes)
        let mut b = Vec::with_capacity(34);
        b.extend_from_slice(&self.sig.to_be_bytes());
        b.extend_from_slice(&self.digest);
        b
    }
}

// ── SigIndex ──────────────────────────────────────────────────────────────────

/// Signature inverted index over a Layer.
pub struct SigIndex {
    /// layer id string for display
    pub layer_name: String,
    /// BTreeMap ensures sorted iteration (deterministic)
    pub map: BTreeMap<u16, PostingList>,
    /// MerkleRoot over all (sig_bytes || posting_digest) entries, sorted by sig
    pub index_digest: [u8; 32],
    /// Total number of elements indexed
    pub total_elems: usize,
}

impl SigIndex {
    /// Build the index from any Layer implementation.
    pub fn build(layer: &dyn Layer) -> Self {
        let n = layer.len();
        let mut acc: BTreeMap<u16, Vec<usize>> = BTreeMap::new();

        for i in 0..n {
            acc.entry(layer.sig(i)).or_default().push(i);
        }

        let map: BTreeMap<u16, PostingList> = acc
            .into_iter()
            .map(|(sig, indices)| (sig, PostingList::build(sig, indices, layer)))
            .collect();

        // index_digest = MerkleRoot over posting list canonical bytes, sorted by sig
        let mut leaves: Vec<[u8; 32]> = map.values()
            .map(|pl| sha256_bytes(&pl.canonical_bytes()))
            .collect();
        leaves.sort_unstable();
        let index_digest = merkle_root(&leaves);

        SigIndex {
            layer_name: layer.id().as_str().to_string(),
            map,
            index_digest,
            total_elems: n,
        }
    }

    /// How many distinct signatures are in the index.
    pub fn sig_count(&self) -> usize {
        self.map.len()
    }

    /// Return the posting list for an exact signature match.
    pub fn get(&self, sig: u16) -> Option<&PostingList> {
        self.map.get(&sig)
    }

    /// Return all posting lists whose sig is within `max_dist` Hamming distance
    /// of `query_sig`, sorted by distance then sig (deterministic).
    pub fn query_hamming(&self, query_sig: u16, max_dist: u32) -> Vec<&PostingList> {
        let mut results: Vec<(u32, u16, &PostingList)> = self.map
            .iter()
            .filter_map(|(&sig, pl)| {
                let d = (sig ^ query_sig).count_ones();
                if d <= max_dist { Some((d, sig, pl)) } else { None }
            })
            .collect();
        results.sort_by(|(da, sa, _), (db, sb, _)| da.cmp(db).then(sa.cmp(sb)));
        results.into_iter().map(|(_, _, pl)| pl).collect()
    }

    /// Find the nearest posting list by Hamming distance.
    /// Ties broken by sig value (deterministic).
    pub fn nearest(&self, query_sig: u16) -> Option<&PostingList> {
        self.map.iter()
            .min_by(|(sa, _), (sb, _)| {
                let da = (*sa ^ query_sig).count_ones();
                let db = (*sb ^ query_sig).count_ones();
                da.cmp(&db).then(sa.cmp(sb))
            })
            .map(|(_, pl)| pl)
    }

    /// Top-k nearest posting lists by Hamming distance.
    pub fn top_k(&self, query_sig: u16, k: usize) -> Vec<&PostingList> {
        let mut scored: Vec<(u32, u16, &PostingList)> = self.map
            .iter()
            .map(|(&sig, pl)| ((sig ^ query_sig).count_ones(), sig, pl))
            .collect();
        scored.sort_by(|(da, sa, _), (db, sb, _)| da.cmp(db).then(sa.cmp(sb)));
        scored.into_iter().take(k).map(|(_, _, pl)| pl).collect()
    }

    /// Verify that a posting list digest matches the index entry.
    /// Returns Ok(()) if valid, Err(msg) if not.
    pub fn verify_posting(&self, sig: u16, claimed_digest: [u8; 32]) -> Result<(), String> {
        match self.map.get(&sig) {
            None => Err(format!("sig {} not in index", sig)),
            Some(pl) => {
                if pl.digest == claimed_digest {
                    Ok(())
                } else {
                    Err(format!(
                        "posting digest mismatch for sig {}: expected {} got {}",
                        sig,
                        hex::encode(pl.digest),
                        hex::encode(claimed_digest)
                    ))
                }
            }
        }
    }

    /// Print a summary of the index for diagnostics.
    pub fn summary(&self) -> String {
        let max_posting = self.map.values().map(|pl| pl.indices.len()).max().unwrap_or(0);
        let avg_posting = if self.map.is_empty() { 0.0 } else {
            self.total_elems as f64 / self.map.len() as f64
        };
        format!(
            "SigIndex[{}]: {} elems, {} distinct sigs, max_posting={}, avg_posting={:.1}, digest={}",
            self.layer_name,
            self.total_elems,
            self.map.len(),
            max_posting,
            avg_posting,
            hex::encode(&self.index_digest[..8])
        )
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::word::WordLayer;
    use crate::syllable::SyllableLayer;
    use crate::semantic::SemanticLayer;
    use crate::layer::Layer;

    #[test]
    fn sig_index_semantic_all_elems_indexed() {
        let layer = SemanticLayer::new();
        let idx = SigIndex::build(&layer);
        let total: usize = idx.map.values().map(|pl| pl.indices.len()).sum();
        assert_eq!(total, layer.len());
    }

    #[test]
    fn sig_index_semantic_digest_stable() {
        let layer = SemanticLayer::new();
        let idx1 = SigIndex::build(&layer);
        let idx2 = SigIndex::build(&layer);
        assert_eq!(idx1.index_digest, idx2.index_digest);
    }

    #[test]
    fn sig_index_semantic_nearest_self() {
        let layer = SemanticLayer::new();
        let idx = SigIndex::build(&layer);
        let s = layer.sig(0);
        let pl = idx.nearest(s).unwrap();
        assert_eq!(pl.sig, s);
    }

    #[test]
    fn sig_index_semantic_verify_posting_ok() {
        let layer = SemanticLayer::new();
        let idx = SigIndex::build(&layer);
        for (sig, pl) in &idx.map {
            assert!(idx.verify_posting(*sig, pl.digest).is_ok());
        }
    }

    #[test]
    fn sig_index_semantic_verify_posting_tamper_detected() {
        let layer = SemanticLayer::new();
        let idx = SigIndex::build(&layer);
        let (&sig, _) = idx.map.iter().next().unwrap();
        let bad_digest = [0xFFu8; 32];
        assert!(idx.verify_posting(sig, bad_digest).is_err());
    }

    #[test]
    fn sig_index_semantic_query_hamming_zero() {
        let layer = SemanticLayer::new();
        let idx = SigIndex::build(&layer);
        let s = layer.sig(0);
        let results = idx.query_hamming(s, 0);
        assert!(!results.is_empty());
        assert!(results.iter().all(|pl| pl.sig == s));
    }

    #[test]
    fn sig_index_semantic_top_k_ordered() {
        let layer = SemanticLayer::new();
        let idx = SigIndex::build(&layer);
        let s = layer.sig(0);
        let k = idx.top_k(s, 3);
        assert!(!k.is_empty());
        if k.len() >= 2 {
            let d0 = (k[0].sig ^ s).count_ones();
            let d1 = (k[1].sig ^ s).count_ones();
            assert!(d0 <= d1);
        }
    }

    #[test]
    fn sig_index_word_all_elems_indexed() {
        let layer = WordLayer::new();
        let idx = SigIndex::build(&layer);
        let total: usize = idx.map.values().map(|pl| pl.indices.len()).sum();
        assert_eq!(total, layer.len());
    }

    #[test]
    fn sig_index_word_digest_stable() {
        let layer = WordLayer::new();
        let idx1 = SigIndex::build(&layer);
        let idx2 = SigIndex::build(&layer);
        assert_eq!(idx1.index_digest, idx2.index_digest);
    }

    #[test]
    fn sig_index_word_sig_count() {
        let layer = WordLayer::new();
        let idx = SigIndex::build(&layer);
        // 8-bit sig → at most 256 distinct values
        assert!(idx.sig_count() <= 256);
        assert!(idx.sig_count() > 0);
    }

    #[test]
    fn sig_index_word_nearest_self() {
        let layer = WordLayer::new();
        let idx = SigIndex::build(&layer);
        let s = layer.sig(0);
        let pl = idx.nearest(s).unwrap();
        assert_eq!((pl.sig ^ s).count_ones(), 0);
    }

    #[test]
    fn sig_index_word_verify_all_postings() {
        let layer = WordLayer::new();
        let idx = SigIndex::build(&layer);
        for (sig, pl) in &idx.map {
            assert!(idx.verify_posting(*sig, pl.digest).is_ok(),
                "posting verify failed for sig {}", sig);
        }
    }

    #[test]
    fn sig_index_syllable_all_elems_indexed() {
        let layer = SyllableLayer::new();
        let idx = SigIndex::build(&layer);
        let total: usize = idx.map.values().map(|pl| pl.indices.len()).sum();
        assert_eq!(total, layer.len());
    }

    #[test]
    fn sig_index_syllable_digest_stable() {
        let layer = SyllableLayer::new();
        let idx1 = SigIndex::build(&layer);
        let idx2 = SigIndex::build(&layer);
        assert_eq!(idx1.index_digest, idx2.index_digest);
    }

    #[test]
    fn sig_index_syllable_sig_count() {
        let layer = SyllableLayer::new();
        let idx = SigIndex::build(&layer);
        // 16-bit sig → up to 65536 distinct values, but expect good coverage
        assert!(idx.sig_count() > 100);
        println!("{}", idx.summary());
    }

    #[test]
    fn sig_index_syllable_nearest_self() {
        let layer = SyllableLayer::new();
        let idx = SigIndex::build(&layer);
        let s = layer.sig(0);
        let pl = idx.nearest(s).unwrap();
        assert_eq!((pl.sig ^ s).count_ones(), 0);
    }

    #[test]
    fn sig_index_posting_canonical_bytes_deterministic() {
        let layer = SemanticLayer::new();
        let idx = SigIndex::build(&layer);
        let pl = idx.map.values().next().unwrap();
        assert_eq!(pl.canonical_bytes(), pl.canonical_bytes());
    }

    #[test]
    fn sig_index_summary_nonempty() {
        let layer = WordLayer::new();
        let idx = SigIndex::build(&layer);
        let s = idx.summary();
        assert!(s.contains("SigIndex[WORD]"));
        assert!(s.contains("elems"));
    }
}
