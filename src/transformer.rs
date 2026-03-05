//! Transformer block assembly — Stage 7 of the tower blueprint.
//!
//! Standard transformer block:
//!   x = x + Attention(LayerNorm(x))
//!   x = x + FFN(LayerNorm(x))
//!
//! This system's block:
//!   context = context + CertifiedAttention(context, active_layer, tau)
//!   context = context + TowerFFN(active_layer → next_layer)(witness)
//!
//! A full forward pass = 6 blocks (PHONEME→DISCOURSE).
//! Each block produces a certified BlockOutput with:
//!   - attention result (within-layer)
//!   - ffn step (cross-layer projection)
//!   - block digest chained from previous block digest
//!
//! The chain of block digests is the verifiable semtrace.
//! Any tampering with any block breaks all subsequent digests.

use crate::digest::sha256_bytes;
use crate::layer::{Layer, LayerId};
use crate::attention::{CertifiedAttention, AttentionResult};
use crate::feedforward::{TowerFFN, FFNStep};

// ── BlockConfig ───────────────────────────────────────────────────────────────

/// Configuration for a single transformer block.
#[derive(Clone, Debug)]
pub struct BlockConfig {
    pub src_layer:    LayerId,
    pub tgt_layer:    LayerId,
    pub tau:          f64,
    pub attn_top_k:   usize,
    pub ffn_top_k:    usize,
}

impl BlockConfig {
    pub fn new(src: LayerId, tgt: LayerId, tau: f64, k: usize) -> Self {
        BlockConfig { src_layer: src, tgt_layer: tgt, tau, attn_top_k: k, ffn_top_k: k }
    }

    /// Default 6-block config for a full upward pass.
    pub fn default_tower_pass(tau: f64, k: usize) -> Vec<BlockConfig> {
        vec![
            BlockConfig::new(LayerId::Phoneme,   LayerId::Syllable,  tau, k),
            BlockConfig::new(LayerId::Syllable,  LayerId::Morpheme,  tau, k),
            BlockConfig::new(LayerId::Morpheme,  LayerId::Word,      tau, k),
            BlockConfig::new(LayerId::Word,      LayerId::Phrase,    tau, k),
            BlockConfig::new(LayerId::Phrase,    LayerId::Semantic,  tau, k),
            BlockConfig::new(LayerId::Semantic,  LayerId::Discourse, tau, k),
        ]
    }
}

// ── BlockOutput ───────────────────────────────────────────────────────────────

/// Output of a single transformer block.
#[derive(Clone, Debug)]
pub struct BlockOutput {
    pub block_idx:      usize,
    pub src_layer:      LayerId,
    pub tgt_layer:      LayerId,
    /// Within-layer attention result (attends on src_layer).
    pub attention:      AttentionResult,
    /// Cross-layer FFN step (projects src → tgt, attends on tgt).
    pub ffn:            FFNStep,
    /// Digest of this block chained with previous block digest.
    pub block_digest:   [u8; 32],
    /// The element index passed forward to the next block.
    pub output_elem_idx: usize,
    /// The layer of the output element.
    pub output_layer:   LayerId,
}

impl BlockOutput {
    pub fn canonical_bytes(&self) -> Vec<u8> {
        format!(
            "{{\"attention_digest\":\"{}\",\"block_idx\":{},\"ffn_digest\":\"{}\",\"src_layer\":\"{}\",\"tgt_layer\":\"{}\"}}",
            hex::encode(self.attention.result_digest),
            self.block_idx,
            hex::encode(self.ffn.step_digest),
            self.src_layer.as_str(),
            self.tgt_layer.as_str(),
        ).into_bytes()
    }

    pub fn verify(&self) -> Result<(), String> {
        self.attention.verify()?;
        self.ffn.verify()?;
        Ok(())
    }
}

// ── ForwardPass ───────────────────────────────────────────────────────────────

/// The complete output of a tower forward pass.
#[derive(Clone, Debug)]
pub struct ForwardPass {
    pub blocks:       Vec<BlockOutput>,
    /// Final element index in the discourse layer.
    pub final_idx:    usize,
    /// SHA-256 chain over all block digests in order.
    pub pass_digest:  [u8; 32],
}

impl ForwardPass {
    pub fn verify_all(&self) -> Result<(), String> {
        for (i, block) in self.blocks.iter().enumerate() {
            block.verify()
                .map_err(|e| format!("block {} failed: {}", i, e))?;
        }
        // Verify pass digest
        let expected = Self::compute_pass_digest(&self.blocks);
        if expected != self.pass_digest {
            return Err(format!(
                "pass_digest mismatch: expected {} got {}",
                hex::encode(expected),
                hex::encode(self.pass_digest),
            ));
        }
        Ok(())
    }

    fn compute_pass_digest(blocks: &[BlockOutput]) -> [u8; 32] {
        let mut chain = sha256_bytes(b"tower_forward_pass_v1");
        for block in blocks {
            let mut combined = Vec::with_capacity(64);
            combined.extend_from_slice(&chain);
            combined.extend_from_slice(&block.block_digest);
            chain = sha256_bytes(&combined);
        }
        chain
    }

    /// Render a human-readable trace of the forward pass.
    pub fn render_trace(&self) -> String {
        let mut lines = vec![
            format!("ForwardPass digest={}", hex::encode(&self.pass_digest[..8])),
        ];
        for block in &self.blocks {
            lines.push(format!(
                "  Block[{}] {}→{} | attn_top={} | ffn_top={} | digest={}",
                block.block_idx,
                block.src_layer.as_str(),
                block.tgt_layer.as_str(),
                block.attention.weights.first()
                    .map(|w| w.rendered.as_str()).unwrap_or("?"),
                block.ffn.attention.weights.first()
                    .map(|w| w.rendered.as_str()).unwrap_or("?"),
                hex::encode(&block.block_digest[..8]),
            ));
        }
        lines.join("\n")
    }
}

// ── TowerTransformer ──────────────────────────────────────────────────────────

pub struct TowerTransformer;

impl TowerTransformer {
    /// Run a single certified transformer block.
    ///
    /// 1. CertifiedAttention on src_layer using query_sig
    /// 2. TowerFFN: project src elem → tgt_layer, attend on tgt_layer
    /// 3. Chain block digest from prev_digest
    pub fn block(
        block_idx:    usize,
        src_layer:    &dyn Layer,
        tgt_layer:    &dyn Layer,
        elem_idx:     usize,
        tau:          f64,
        attn_top_k:   usize,
        ffn_top_k:    usize,
        prev_digest:  &[u8; 32],
    ) -> Result<BlockOutput, String> {
        let query_sig = src_layer.sig(elem_idx);

        // Sublayer 1: within-layer certified attention
        let attention = CertifiedAttention::attend(
            src_layer,
            query_sig,
            tau,
            attn_top_k,
        );

        // Sublayer 2: cross-layer certified FFN
        let ffn = TowerFFN::forward(src_layer, tgt_layer, elem_idx, tau, ffn_top_k)?;

        // Output element: top of FFN attention on tgt_layer
        let output_elem_idx = ffn.attention.top_idx;

        // Chain digest: SHA-256(prev_digest || block_canonical)
        let mut block_out = BlockOutput {
            block_idx,
            src_layer:       src_layer.id(),
            tgt_layer:       tgt_layer.id(),
            attention,
            ffn,
            block_digest:    [0u8; 32],
            output_elem_idx,
            output_layer:    tgt_layer.id(),
        };
        let block_canonical = block_out.canonical_bytes();
        let mut combined = Vec::with_capacity(64);
        combined.extend_from_slice(prev_digest);
        combined.extend_from_slice(&sha256_bytes(&block_canonical));
        block_out.block_digest = sha256_bytes(&combined);

        Ok(block_out)
    }

    /// Run a full certified forward pass through the tower.
    pub fn forward_pass(
        start_elem_idx: usize,
        phoneme_layer:  &dyn Layer,
        syllable_layer: &dyn Layer,
        morpheme_layer: &dyn Layer,
        word_layer:     &dyn Layer,
        phrase_layer:   &dyn Layer,
        semantic_layer: &dyn Layer,
        discourse_layer:&dyn Layer,
        tau:            f64,
        top_k:          usize,
    ) -> Result<ForwardPass, String> {
        let layers: &[(&dyn Layer, &dyn Layer)] = &[
            (phoneme_layer,  syllable_layer),
            (syllable_layer, morpheme_layer),
            (morpheme_layer, word_layer),
            (word_layer,     phrase_layer),
            (phrase_layer,   semantic_layer),
            (semantic_layer, discourse_layer),
        ];

        let mut blocks  = Vec::with_capacity(6);
        let mut elem_idx = start_elem_idx;
        let mut prev_digest = sha256_bytes(b"tower_genesis");

        for (i, &(src, tgt)) in layers.iter().enumerate() {
            let block = Self::block(
                i, src, tgt, elem_idx,
                tau, top_k, top_k,
                &prev_digest,
            )?;
            prev_digest = block.block_digest;
            elem_idx    = block.output_elem_idx;
            blocks.push(block);
        }

        let final_idx   = elem_idx;
        let pass_digest = ForwardPass::compute_pass_digest(&blocks);

        Ok(ForwardPass { blocks, final_idx, pass_digest })
    }

    /// Run forward passes for multiple starting elements and return all results.
    pub fn batch_forward(
        start_indices:  &[usize],
        phoneme_layer:  &dyn Layer,
        syllable_layer: &dyn Layer,
        morpheme_layer: &dyn Layer,
        word_layer:     &dyn Layer,
        phrase_layer:   &dyn Layer,
        semantic_layer: &dyn Layer,
        discourse_layer:&dyn Layer,
        tau:            f64,
        top_k:          usize,
    ) -> Vec<Result<ForwardPass, String>> {
        start_indices.iter().map(|&idx| {
            Self::forward_pass(
                idx,
                phoneme_layer, syllable_layer, morpheme_layer,
                word_layer, phrase_layer, semantic_layer, discourse_layer,
                tau, top_k,
            )
        }).collect()
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
    fn block_config_default_tower_pass_len() {
        let configs = BlockConfig::default_tower_pass(1.0, 5);
        assert_eq!(configs.len(), 6);
    }

    #[test]
    fn block_config_layers_in_order() {
        let configs = BlockConfig::default_tower_pass(1.0, 5);
        assert_eq!(configs[0].src_layer, LayerId::Phoneme);
        assert_eq!(configs[0].tgt_layer, LayerId::Syllable);
        assert_eq!(configs[5].src_layer, LayerId::Semantic);
        assert_eq!(configs[5].tgt_layer, LayerId::Discourse);
    }

    #[test]
    fn single_block_verify() {
        let (ph, syl, ..) = all_layers();
        let genesis = sha256_bytes(b"tower_genesis");
        let block = TowerTransformer::block(
            0, &ph, &syl, 0, 1.0, 5, 5, &genesis
        ).unwrap();
        block.verify().unwrap();
    }

    #[test]
    fn single_block_digest_stable() {
        let (ph, syl, ..) = all_layers();
        let genesis = sha256_bytes(b"tower_genesis");
        let b1 = TowerTransformer::block(0, &ph, &syl, 0, 1.0, 5, 5, &genesis).unwrap();
        let b2 = TowerTransformer::block(0, &ph, &syl, 0, 1.0, 5, 5, &genesis).unwrap();
        assert_eq!(b1.block_digest, b2.block_digest);
    }

    #[test]
    fn single_block_digest_chained() {
        let (ph, syl, ..) = all_layers();
        let genesis1 = sha256_bytes(b"tower_genesis");
        let genesis2 = sha256_bytes(b"different_genesis");
        let b1 = TowerTransformer::block(0, &ph, &syl, 0, 1.0, 5, 5, &genesis1).unwrap();
        let b2 = TowerTransformer::block(0, &ph, &syl, 0, 1.0, 5, 5, &genesis2).unwrap();
        // Different prev_digest → different block_digest
        assert_ne!(b1.block_digest, b2.block_digest);
    }

    #[test]
    fn forward_pass_block_count() {
        let (ph, syl, morph, word, phrase, sem, disc) = all_layers();
        let pass = TowerTransformer::forward_pass(
            0, &ph, &syl, &morph, &word, &phrase, &sem, &disc, 1.0, 3
        ).unwrap();
        assert_eq!(pass.blocks.len(), 6);
    }

    #[test]
    fn forward_pass_verify_all() {
        let (ph, syl, morph, word, phrase, sem, disc) = all_layers();
        let pass = TowerTransformer::forward_pass(
            0, &ph, &syl, &morph, &word, &phrase, &sem, &disc, 1.0, 3
        ).unwrap();
        pass.verify_all().unwrap();
    }

    #[test]
    fn forward_pass_digest_stable() {
        let (ph, syl, morph, word, phrase, sem, disc) = all_layers();
        let p1 = TowerTransformer::forward_pass(
            0, &ph, &syl, &morph, &word, &phrase, &sem, &disc, 1.0, 3
        ).unwrap();
        let p2 = TowerTransformer::forward_pass(
            0, &ph, &syl, &morph, &word, &phrase, &sem, &disc, 1.0, 3
        ).unwrap();
        assert_eq!(p1.pass_digest, p2.pass_digest);
    }

    #[test]
    fn forward_pass_digest_differs_by_start() {
        let (ph, syl, morph, word, phrase, sem, disc) = all_layers();
        let p0 = TowerTransformer::forward_pass(
            0, &ph, &syl, &morph, &word, &phrase, &sem, &disc, 1.0, 3
        ).unwrap();
        let p1 = TowerTransformer::forward_pass(
            1, &ph, &syl, &morph, &word, &phrase, &sem, &disc, 1.0, 3
        ).unwrap();
        assert_ne!(p0.pass_digest, p1.pass_digest);
    }

    #[test]
    fn forward_pass_block_digests_chained() {
        let (ph, syl, morph, word, phrase, sem, disc) = all_layers();
        let pass = TowerTransformer::forward_pass(
            0, &ph, &syl, &morph, &word, &phrase, &sem, &disc, 1.0, 3
        ).unwrap();
        // Each block digest must differ from the previous
        for i in 1..pass.blocks.len() {
            assert_ne!(pass.blocks[i].block_digest, pass.blocks[i-1].block_digest,
                "blocks {} and {} have same digest", i-1, i);
        }
    }

    #[test]
    fn forward_pass_render_trace_nonempty() {
        let (ph, syl, morph, word, phrase, sem, disc) = all_layers();
        let pass = TowerTransformer::forward_pass(
            0, &ph, &syl, &morph, &word, &phrase, &sem, &disc, 1.0, 3
        ).unwrap();
        let trace = pass.render_trace();
        assert!(trace.contains("ForwardPass"));
        assert!(trace.contains("PHONEME"));
        assert!(trace.contains("DISCOURSE"));
    }

    #[test]
    fn forward_pass_output_in_discourse() {
        let (ph, syl, morph, word, phrase, sem, disc) = all_layers();
        let pass = TowerTransformer::forward_pass(
            0, &ph, &syl, &morph, &word, &phrase, &sem, &disc, 1.0, 3
        ).unwrap();
        assert_eq!(pass.blocks.last().unwrap().output_layer, LayerId::Discourse);
        assert!(pass.final_idx < disc.len());
    }

    #[test]
    fn batch_forward_all_succeed() {
        let (ph, syl, morph, word, phrase, sem, disc) = all_layers();
        let results = TowerTransformer::batch_forward(
            &[0, 1, 2],
            &ph, &syl, &morph, &word, &phrase, &sem, &disc, 1.0, 3
        );
        assert_eq!(results.len(), 3);
        for r in &results {
            assert!(r.is_ok(), "batch forward failed: {:?}", r);
        }
    }

    #[test]
    fn batch_forward_digests_differ() {
        let (ph, syl, morph, word, phrase, sem, disc) = all_layers();
        let results = TowerTransformer::batch_forward(
            &[0, 1, 2],
            &ph, &syl, &morph, &word, &phrase, &sem, &disc, 1.0, 3
        );
        let digests: Vec<_> = results.iter()
            .map(|r| r.as_ref().unwrap().pass_digest)
            .collect();
        // All three should be different (different starting phonemes)
        assert_ne!(digests[0], digests[1]);
        assert_ne!(digests[1], digests[2]);
    }
}
