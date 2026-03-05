//! run_onnx_proposer — Stage 18 demo binary.
//!
//! Runs the ONNX learned proposer through the full 12-op canonical sequence
//! using REAL step digests from the certified tower forward pass.
//! This is true end-to-end Rust inference: tower → real context → ONNX → op.
//!
//! Usage:
//!   cargo run --bin run_onnx_proposer -- [--n 5] [--model train/model.onnx]

use anyhow::{Context, Result};
use llm_nature_semantic_transformer::{
    layer::{Layer, LayerId},
    onnx_proposer::OnnxProposer,
    proposer::{OpKind, ProposerContext},
    tower::Tower,
};

// Ground-truth canonical op sequence (from corpus_v2.ndjson)
const GT_OPS: &[OpKind] = &[
    OpKind::SelectUniverse,
    OpKind::WitnessNearest,
    OpKind::Attend,
    OpKind::FFNStep,   // PHONEME   -> SYLLABLE
    OpKind::FFNStep,   // SYLLABLE  -> MORPHEME
    OpKind::FFNStep,   // MORPHEME  -> WORD
    OpKind::FFNStep,   // WORD      -> PHRASE
    OpKind::FFNStep,   // PHRASE    -> SEMANTIC
    OpKind::FFNStep,   // SEMANTIC  -> DISCOURSE
    OpKind::ProjectLayer,
    OpKind::ReturnSet,
    OpKind::Accept,
];

// Canonical active layer at each block position
const CANONICAL_LAYERS: &[LayerId] = &[
    LayerId::Phoneme,   // 0  SELECT_UNIVERSE
    LayerId::Phoneme,   // 1  WITNESS_NEAREST
    LayerId::Phoneme,   // 2  ATTEND
    LayerId::Phoneme,   // 3  FFN_STEP  (PHONEME->SYLLABLE)
    LayerId::Syllable,  // 4  FFN_STEP  (SYLLABLE->MORPHEME)
    LayerId::Morpheme,  // 5  FFN_STEP  (MORPHEME->WORD)
    LayerId::Word,      // 6  FFN_STEP  (WORD->PHRASE)
    LayerId::Phrase,    // 7  FFN_STEP  (PHRASE->SEMANTIC)
    LayerId::Discourse, // 8  FFN_STEP  (SEMANTIC->DISCOURSE) — output layer is DISCOURSE
    LayerId::Discourse, // 9  PROJECT_LAYER
    LayerId::Discourse, // 10 RETURN_SET
    LayerId::Discourse, // 11 ACCEPT
];

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let n_phonemes = args.windows(2)
        .find(|w| w[0] == "--n")
        .and_then(|w| w[1].parse::<usize>().ok())
        .unwrap_or(5);
    let model_path = args.windows(2)
        .find(|w| w[0] == "--model")
        .map(|w| w[1].clone())
        .unwrap_or_else(|| "train/model.onnx".to_string());

    println!("Building tower...");
    let tower = Tower::build();
    println!("Tower root:  {}", hex::encode(&tower.manifest.root_digest[..8]));
    println!("Phonemes:    {}", tower.phoneme.len());
    println!("Model:       {model_path}");
    println!();

    let mut proposer = OnnxProposer::new(&model_path)
        .with_context(|| format!("failed to load {model_path}"))?;

    let mut correct = 0usize;
    let mut total   = 0usize;

    for ph_idx in 0..n_phonemes.min(tower.phoneme.len()) {
        let ph_name = tower.phoneme.render(ph_idx);

        // Run a REAL certified forward pass to get real block digests
        let pass = tower.forward(ph_idx, 1.0, 3)
            .map_err(|e| anyhow::anyhow!("forward pass failed: {e}"))?;

        println!("── Phoneme {ph_idx}: {ph_name}  pass={}",
            hex::encode(&pass.pass_digest[..8]));
        println!("  {:>3}  {:>20}  {:>8}  {:>20}", "blk", "ONNX", "vs GT", "ground truth");

        // Build step digest array from real forward pass
        // blocks[0..5] = FFN blocks (PHONEME->SYLLABLE .. SEMANTIC->DISCOURSE)
        // For blocks 0-2 (SELECT, WITNESS, ATTEND) use attention digest from block 0
        // For blocks 3-8 (FFN_STEPs) use block[i-3].ffn.step_digest
        // For blocks 9-11 use pass_digest
        let step_digests: Vec<[u8; 32]> = (0..12).map(|bi| {
            match bi {
                0..=2 => pass.blocks[0].attention.result_digest, // pre-FFN ops
                3     => pass.blocks[0].ffn.step_digest,          // PHONEME->SYLLABLE
                4     => pass.blocks[1].ffn.step_digest,          // SYLLABLE->MORPHEME
                5     => pass.blocks[2].ffn.step_digest,          // MORPHEME->WORD
                6     => pass.blocks[3].ffn.step_digest,          // WORD->PHRASE
                7     => pass.blocks[4].ffn.step_digest,          // PHRASE->SEMANTIC
                8     => pass.blocks[5].ffn.step_digest,          // SEMANTIC->DISCOURSE
                _     => pass.pass_digest,                         // PROJECT, RETURN, ACCEPT
            }
        }).collect();

        let mut ctx = ProposerContext::new(LayerId::Phoneme);

        for block_idx in 0..12usize {
            let gt_op  = GT_OPS[block_idx];
            let dist   = proposer.propose(&ctx, block_idx, 1.0, 3)?;
            let top_op = dist.top().map(|o| o.kind).unwrap_or(OpKind::Reject);
            let ok     = top_op == gt_op;

            if ok { correct += 1; }
            total += 1;

            println!("  {:>3}  {:>20}  {:>8}  {:>20}",
                block_idx,
                top_op.as_str(),
                if ok { "✓" } else { "✗" },
                gt_op.as_str());

            // Advance context with REAL step digest
            ctx.advance(&step_digests[block_idx], CANONICAL_LAYERS[block_idx]);
        }
        println!();
    }

    let pct = 100.0 * correct as f64 / total.max(1) as f64;
    println!("Accuracy vs ground truth: {correct}/{total} ({pct:.1}%)");

    // Verify the forward passes
    println!("\nVerifying forward passes...");
    let mut verified = 0usize;
    for ph_idx in 0..n_phonemes.min(tower.phoneme.len()) {
        let pass = tower.forward(ph_idx, 1.0, 3)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        match pass.verify_all() {
            Ok(_)  => { verified += 1; print!("✓ "); }
            Err(e) => print!("✗({e}) "),
        }
    }
    println!("\nForward pass verification: {verified}/{n_phonemes}");

    let gate = correct == total;
    println!("\nStage 18 gate (ONNX 100% vs GT with real digests): {}",
        if gate { "✓ PASSED" } else { "✗ FAILED" });

    Ok(())
}