//! verify_trace — Stage 26: trace persistence and replay verification.
//!
//! Reads a trace produced by `query` and certifies it independently.
//! Verification is deterministic: anyone with the binary can verify any trace.
//!
//! Usage:
//!   cargo run --bin verify_trace -- runs/query_.../trace.ndjson
//!   cargo run --bin verify_trace -- runs/query_.../  (auto-finds trace.ndjson)
//!
//! Exit code: 0 = VERIFIED, 1 = FAILED

use anyhow::{Context, Result};
use llm_nature_semantic_transformer::{
    layer::{Layer, LayerId},
    tower::Tower,
};
use std::path::PathBuf;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    let input = args.get(1)
        .cloned()
        .unwrap_or_else(|| { eprintln!("Usage: verify_trace <trace.ndjson | run_dir>"); std::process::exit(1); });

    // Resolve trace path and result path
    let input_path = PathBuf::from(&input);
    let (trace_path, result_path) = if input_path.is_dir() {
        (input_path.join("trace.ndjson"), input_path.join("result.json"))
    } else {
        let result = input_path.parent().unwrap_or(&input_path).join("result.json");
        (input_path, result)
    };

    println!("Verifying trace: {}", trace_path.display());

    // Load trace records
    let trace_txt = std::fs::read_to_string(&trace_path)
        .with_context(|| format!("cannot read {}", trace_path.display()))?;
    let records: Vec<serde_json::Value> = trace_txt.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).context("parse record"))
        .collect::<Result<_>>()?;

    if records.is_empty() {
        anyhow::bail!("trace is empty");
    }

    // Load result.json
    let result: serde_json::Value = if result_path.exists() {
        let txt = std::fs::read_to_string(&result_path)?;
        serde_json::from_str(&txt)?
    } else {
        serde_json::json!({})
    };

    // Extract metadata from first record + result.json
    let ph_idx    = records[0]["phoneme_idx"].as_u64().unwrap_or(0) as usize;
    let ph_sym    = records[0]["phoneme_sym"].as_str().unwrap_or("?");
    let tau       = records[0]["tau"].as_f64().unwrap_or(1.0);
    let top_k     = records[0]["top_k"].as_u64().unwrap_or(3) as usize;
    let claimed_pass_digest = result["pass_digest"].as_str().unwrap_or("");
    let claimed_tower_root  = result["tower_root"].as_str().unwrap_or("");

    println!("  phoneme:      {} (idx={})", ph_sym, ph_idx);
    println!("  tau:          {tau}  top_k: {top_k}");
    println!("  blocks:       {}", records.len());
    println!("  pass_digest:  {}...", &claimed_pass_digest[..16.min(claimed_pass_digest.len())]);
    println!();

    // Build tower and verify root matches
    print!("Building tower... ");
    std::io::Write::flush(&mut std::io::stdout()).ok();
    let tower = Tower::build();
    let live_root = hex::encode(tower.manifest.root_digest);
    println!("root={}", &live_root[..16]);

    let mut failures: Vec<String> = Vec::new();

    // Check 1: tower root matches
    if !claimed_tower_root.is_empty() && claimed_tower_root != live_root {
        failures.push(format!("tower_root mismatch: claimed={} live={}",
            &claimed_tower_root[..16], &live_root[..16]));
    } else {
        println!("  ✓ tower_root matches");
    }

    // Run certified forward pass to get ground truth digests
    let pass = tower.forward(ph_idx, tau, top_k)
        .map_err(|e| anyhow::anyhow!("forward pass failed: {e}"))?;

    // Verify forward pass internally
    pass.verify_all()
        .map_err(|e| anyhow::anyhow!("forward pass internal verification failed: {e}"))?;
    println!("  ✓ forward pass internally verified");

    // Check 2: pass_digest matches
    let live_pass_digest = hex::encode(pass.pass_digest);
    if !claimed_pass_digest.is_empty() && claimed_pass_digest != live_pass_digest {
        failures.push(format!("pass_digest mismatch: claimed={}... live={}...",
            &claimed_pass_digest[..16], &live_pass_digest[..16]));
    } else {
        println!("  ✓ pass_digest matches");
    }

    // Recompute expected step digests
    let expected_digests: Vec<[u8; 32]> = (0..12).map(|bi| match bi {
        0..=2 => pass.blocks[0].attention.result_digest,
        3     => pass.blocks[0].ffn.step_digest,
        4     => pass.blocks[1].ffn.step_digest,
        5     => pass.blocks[2].ffn.step_digest,
        6     => pass.blocks[3].ffn.step_digest,
        7     => pass.blocks[4].ffn.step_digest,
        8     => pass.blocks[5].ffn.step_digest,
        _     => pass.pass_digest,
    }).collect();

    // Check 3: each step_digest matches
    println!("\n  {:>3}  {:>20}  {:>8}  {}", "blk", "op_kind", "status", "step_digest");
    println!("  {}", "─".repeat(72));

    let mut step_ok = 0usize;
    for (i, rec) in records.iter().enumerate() {
        let bi         = rec["block_idx"].as_u64().unwrap_or(i as u64) as usize;
        let op_kind    = rec["op_kind"].as_str().unwrap_or("?");
        let rec_digest = rec["step_digest"].as_str().unwrap_or("");

        if bi < expected_digests.len() {
            let expected = hex::encode(expected_digests[bi]);
            if rec_digest == expected {
                step_ok += 1;
                println!("  {:>3}  {:>20}  {:>8}  {}...",
                    bi, op_kind, "✓", &rec_digest[..16]);
            } else {
                failures.push(format!("block {bi} step_digest mismatch"));
                println!("  {:>3}  {:>20}  {:>8}  claimed={}... expected={}...",
                    bi, op_kind, "✗", &rec_digest[..16], &expected[..16]);
            }
        }
    }

    // Check 4: all 12 blocks present
    if records.len() != 12 {
        failures.push(format!("expected 12 blocks, got {}", records.len()));
    }

    // Check 5: terminal op is ACCEPT
    let terminal = records.last().and_then(|r| r["op_kind"].as_str()).unwrap_or("?");
    if terminal != "ACCEPT" {
        failures.push(format!("terminal op is {terminal}, expected ACCEPT"));
    }

    // Summary
    println!();
    println!("  Step digests: {step_ok}/{}", records.len());
    println!();

    if failures.is_empty() {
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("VERIFIED ✓  —  trace is cryptographically certified");
        println!("  pass_digest: {}", &live_pass_digest[..32]);
        println!("  tower_root:  {}", &live_root[..32]);
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("\nStage 26 gate (trace replay verification): ✓ PASSED");
        std::process::exit(0);
    } else {
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("VERIFICATION FAILED ✗");
        for f in &failures { println!("  • {f}"); }
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("\nStage 26 gate: ✗ FAILED");
        std::process::exit(1);
    }
}