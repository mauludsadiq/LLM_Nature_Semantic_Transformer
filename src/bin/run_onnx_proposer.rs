//! run_onnx_proposer — Stage 18 demo binary.
//!
//! Validates the ONNX proposer against the canonical ground-truth sequence,
//! then collects a corpus and compares record counts with the rule-based proposer.
//!
//! Usage:
//!   cargo run --bin run_onnx_proposer -- [--n 5] [--model train/model.onnx]

use anyhow::{Context, Result};
use llm_nature_semantic_transformer::{
    layer::{Layer, LayerId},
    onnx_proposer::OnnxProposer,
    proposer::{OpKind, ProposerContext, RuleBasedProposer},
    tower::Tower,
};

// Ground-truth canonical sequence from corpus_v2.ndjson
const GT_OPS: &[OpKind] = &[
    OpKind::SelectUniverse,
    OpKind::WitnessNearest,
    OpKind::Attend,
    OpKind::FFNStep,
    OpKind::FFNStep,
    OpKind::FFNStep,
    OpKind::FFNStep,
    OpKind::FFNStep,
    OpKind::FFNStep,
    OpKind::ProjectLayer,
    OpKind::ReturnSet,
    OpKind::Accept,
];

const CANONICAL_LAYERS: &[LayerId] = &[
    LayerId::Phoneme, LayerId::Phoneme, LayerId::Phoneme,
    LayerId::Phoneme, LayerId::Syllable, LayerId::Morpheme,
    LayerId::Word, LayerId::Phrase, LayerId::Semantic,
    LayerId::Discourse, LayerId::Discourse, LayerId::Discourse,
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

    let mut onnx_correct = 0usize;
    let mut rule_correct = 0usize;
    let mut total        = 0usize;

    for ph_idx in 0..n_phonemes.min(tower.phoneme.len()) {
        let ph_name = tower.phoneme.render(ph_idx);
        println!("── Phoneme {ph_idx}: {ph_name}");
        println!("  {:>3}  {:>20}  {:>8}  {:>20}  {:>8}  {:>10}",
            "blk", "ONNX", "vs GT", "Rule", "vs GT", "ground truth");

        let mut ctx_onnx = ProposerContext::new(LayerId::Phoneme);
        let mut ctx_rule = ProposerContext::new(LayerId::Phoneme);

        for block_idx in 0..12usize {
            let gt_op = GT_OPS[block_idx];

            let dist_onnx = proposer.propose(&ctx_onnx, block_idx, 1.0, 3)?;
            let op_onnx   = dist_onnx.top().map(|o| o.kind).unwrap_or(OpKind::Reject);

            let dist_rule = RuleBasedProposer::propose(&ctx_rule);
            let op_rule   = dist_rule.top().map(|o| o.kind).unwrap_or(OpKind::Reject);

            let onnx_ok = op_onnx == gt_op;
            let rule_ok = op_rule == gt_op;
            if onnx_ok { onnx_correct += 1; }
            if rule_ok { rule_correct += 1; }
            total += 1;

            println!("  {:>3}  {:>20}  {:>8}  {:>20}  {:>8}  {:>10}",
                block_idx,
                op_onnx.as_str(), if onnx_ok { "✓" } else { "✗" },
                op_rule.as_str(), if rule_ok { "✓" } else { "✗" },
                gt_op.as_str());

            let step_digest = [0u8; 32];
            ctx_onnx.advance(&step_digest, CANONICAL_LAYERS[block_idx]);
            ctx_rule.advance(&step_digest, CANONICAL_LAYERS[block_idx]);
        }
        println!();
    }

    println!("Accuracy vs ground truth:");
    println!("  ONNX:  {onnx_correct}/{total} ({:.1}%)",
        100.0 * onnx_correct as f64 / total.max(1) as f64);
    println!("  Rule:  {rule_correct}/{total} ({:.1}%)",
        100.0 * rule_correct as f64 / total.max(1) as f64);

    // Corpus comparison
    println!("\nCollecting corpora ({n_phonemes} phonemes)...");
    let onnx_trainer = tower.collect_trace_corpus_onnx(n_phonemes, 1.0, 3, &model_path)
        .map_err(|e| anyhow::anyhow!(e))?;
    let rule_trainer = tower.collect_trace_corpus(n_phonemes, 1.0, 3);

    println!("  ONNX  records={:<5} accept_rate={:.3}  digest={}",
        onnx_trainer.records.len(),
        onnx_trainer.acceptance_rate(),
        hex::encode(&onnx_trainer.corpus_digest()[..8]));
    println!("  Rule  records={:<5} accept_rate={:.3}  digest={}",
        rule_trainer.records.len(),
        rule_trainer.acceptance_rate(),
        hex::encode(&rule_trainer.corpus_digest()[..8]));

    let gate = onnx_correct == total;
    println!("\nStage 18 gate (ONNX 100% vs GT): {}", if gate { "✓ PASSED" } else { "✗ FAILED" });

    Ok(())
}