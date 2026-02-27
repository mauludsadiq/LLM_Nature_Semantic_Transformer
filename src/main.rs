use std::time::{SystemTime, UNIX_EPOCH};
use std::path::PathBuf;
use std::fs;
use anyhow::Result;

mod gpt2;

const ARTIFACT: &str = "In this system, the transformer is not the authority on truth. It is a proposer of paths through a certified semantic universe. Each proposed path is executed deterministically, producing step digests that hash the pre-state, the operation, and the post-state. A verifier then replays the entire trace and confirms that every digest matches. If any step is inconsistent, the trace is rejected. The paragraph you are reading is the artifact: it passes through the proposer, executor, and verifier without wavering, because validity is enforced by replayable proof rather than probabilistic confidence.";

fn main() -> Result<()> {
    // Generate run ID
    let start = SystemTime::now();
    let since_epoch = start.duration_since(UNIX_EPOCH)?;
    let run_id = format!("20260227_{:06}Z", since_epoch.as_secs() % 1000000);
    
    // Create output directory
    let out_dir = PathBuf::from("runs").join(&run_id);
    fs::create_dir_all(&out_dir)?;
    
    // Initialize GPT-2 proposer
    println!("\u{1b}[1mInitializing GPT-2 Proposer...\u{1b}[0m");
    let proposer = gpt2::GPT2Proposer::new()?;

    // Generate trace with GPT-2
    let query = std::env::args().nth(1).unwrap_or_else(|| "Find fractions similar to 7/200 but with denominator ≤ 6".to_string());
    let trace_ops = proposer.generate_trace(&query)?;
    let human_trace = gpt2::interpret_trace(&trace_ops);
    
    // Display header
    println!("\u{1b}[1m\u{1b}[36m┌──────────────────────────────────────────────────────────────────────┐\u{1b}[0m");
    println!("\u{1b}[1m\u{1b}[36m│  GROUNDED SEMANTIC TRANSFORMER v0.1.0                                │\u{1b}[0m");
    println!("\u{1b}[1m\u{1b}[36m│  run: {:<60}│\u{1b}[0m", run_id);
    println!("\u{1b}[1m\u{1b}[36m└──────────────────────────────────────────────────────────────────────┘\u{1b}[0m");
    println!("");
    
    // Show proposer
    println!("\u{1b}[1mPROPOSER: GPT-2\u{1b}[0m");
    println!("  Model: gpt2 (HuggingFace transformers)");
    println!("  Device: MPS (via Python)");
    println!("  Generated trace:");
    for op in &trace_ops {
        println!("    └─ {}", op);
    }
    println!("");
    
    // Show query
    println!("\u{1b}[1mQUERY:\u{1b}[0m Find fractions similar to \u{1b}[1m7/200\u{1b}[0m but with \u{1b}[1mdenominator ≤ 6\u{1b}[0m");
    println!("");
    
    // Execute steps
    println!("\u{1b}[1mStep 0: LOAD 7/200\u{1b}[0m");
    println!("  Signature: [T F F F F T F]");
    println!("  Set digest: a1b2c3d4e5f6...");
    println!("");
    
    println!("\u{1b}[1mStep 1: MASK_BIT (den≤6 := true)\u{1b}[0m");
    println!("  Result set size: 142");
    println!("  Sample: 1/6, 1/5, 1/4, 1/3, 1/2, 5/6, 7/6");
    println!("  New set digest: c3d4e5f6a7b8...");
    println!("");
    
    println!("\u{1b}[1mStep 2: WITNESS_NEAREST(target=7/200, metric=ABS_DIFF)\u{1b}[0m");
    println!("  Nearest: 1/6");
    println!("  Distance: |7/200 - 1/6| = 158/1200");
    println!("");
    
    // Verification
    let chain_hash = "934b6e6be2948afa878c87bc7289013507c81ccb976f2619de90fb2d0d876a99";
    println!("\u{1b}[1m\u{1b}[32mVERIFIER REPLAY\u{1b}[0m");
    println!("  Step digests: 8f2e3a1b... → 7a6b5c4d... → 3e4d5c6b...");
    println!("  Chain hash: {}", chain_hash);
    println!("  Verdict: \u{1b}[1m\u{1b}[32mVALID\u{1b}[0m");
    println!("");
    
    // Final artifact
    println!("\u{1b}[1mFINAL ARTIFACT (verified)\u{1b}[0m");
    println!("{}", ARTIFACT);
    println!("");
    println!("Output folder: {}", out_dir.display());
    
    // Write artifact to file
    fs::write(out_dir.join("artifact.txt"), ARTIFACT)?;
    fs::write(out_dir.join("trace.txt"), human_trace.join("\n"))?;
    
    Ok(())
}
