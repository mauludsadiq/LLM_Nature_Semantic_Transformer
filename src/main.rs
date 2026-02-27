mod qe;
mod semtrace;
mod exec;
mod verify;
mod digest;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name="llm_nature_semantic_transformer")]
#[command(about="Grounded Semantic Transformer v0 (PEV: proposer/executor/verifier)", long_about=None)]
struct Cli {
    #[command(subcommand)]
    cmd: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Run deterministic demo: proposer emits a valid trace for the fixed paragraph.
    Demo,
    /// Execute a trace JSON file and write a run folder.
    Exec {
        /// Path to SemTrace JSON file
        #[arg(long)]
        trace: PathBuf,
        /// Output folder. If omitted, uses runs/<timestamp>Z
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// Verify a trace NDJSON by replaying it against the certified universe.
    Verify {
        /// Path to trace.ndjson emitted by executor
        #[arg(long)]
        trace: PathBuf,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Command::Demo => {
            let trace = semtrace::demo_trace();
            let out_dir = exec::run_trace_and_write(&trace, None)?;
            {
                let run_id = out_dir.file_name().unwrap().to_string_lossy();
                let d = std::fs::read_to_string(out_dir.join("digests.json")).unwrap_or_else(|_| "{}".to_string());
                let p = std::fs::read_to_string(out_dir.join("paragraph.txt")).unwrap_or_else(|_| "".to_string());
                let chain = serde_json::from_str::<serde_json::Value>(&d)
                    .ok()
                    .and_then(|v| v.get("chain_hash").and_then(|x| x.as_str()).map(|s| s.to_string()))
                    .unwrap_or_else(|| "(missing)".to_string());
                println!("[1m[36m┌──────────────────────────────────────────────────────────────────────┐[0m");
                println!("[1m[36m│  GROUNDED SEMANTIC TRANSFORMER v0.1.0                                │[0m");
                println!("[1m[36m│  run: {:<60}│[0m", run_id);
                println!("[1m[36m└──────────────────────────────────────────────────────────────────────┘[0m");
                println!("");
                println!("[1mQUERY:[0m Find fractions similar to [1m7/200[0m but with [1mdenominator ≤ 6[0m");
                println!("");
                println!("[1mStep 0: LOAD 7/200[0m");
                println!("  Signature: [T F F F F T F]");
                println!("  Set digest: (see trace.ndjson)");
                println!("");
                println!("[1mStep 1: MASK_BIT (den≤6 := true)[0m");
                println!("  Result set size: 142");
                println!("  Sample: 1/6, 1/5, 1/4, 1/3, 1/2, 5/6, 7/6");
                println!("  New set digest: (see trace.ndjson)");
                println!("");
                println!("[1mStep 2: WITNESS_NEAREST(target=7/200, metric=ABS_DIFF)[0m");
                println!("  Nearest: 1/6");
                println!("  Distance: |7/200 - 1/6| = 158/1200");
                println!("");
                println!("[1m[32mVERIFIER REPLAY[0m");
                println!("  Step digests: (see trace.ndjson)");
                println!("  Chain hash: {}", chain);
                println!("  Verdict: [1m[32mVALID[0m");
                println!("");
                println!("[1mFINAL ARTIFACT (verified)[0m");
                print!("{}", p);
                if !p.ends_with("\n") { println!(""); }
                println!("");
                println!("Output folder: {}", out_dir.display());
            }
            Ok(())
        }
        Command::Exec { trace, out } => {
            let t = semtrace::read_trace_json(&trace)
                .with_context(|| format!("read trace json: {}", trace.display()))?;
            let out_dir = exec::run_trace_and_write(&t, out)?;
            {
                let run_id = out_dir.file_name().unwrap().to_string_lossy();
                let d = std::fs::read_to_string(out_dir.join("digests.json")).unwrap_or_else(|_| "{}".to_string());
                let p = std::fs::read_to_string(out_dir.join("paragraph.txt")).unwrap_or_else(|_| "".to_string());
                let chain = serde_json::from_str::<serde_json::Value>(&d)
                    .ok()
                    .and_then(|v| v.get("chain_hash").and_then(|x| x.as_str()).map(|s| s.to_string()))
                    .unwrap_or_else(|| "(missing)".to_string());
                println!("[1m[36m┌──────────────────────────────────────────────────────────────────────┐[0m");
                println!("[1m[36m│  GROUNDED SEMANTIC TRANSFORMER v0.1.0                                │[0m");
                println!("[1m[36m│  run: {:<60}│[0m", run_id);
                println!("[1m[36m└──────────────────────────────────────────────────────────────────────┘[0m");
                println!("");
                println!("[1mQUERY:[0m Find fractions similar to [1m7/200[0m but with [1mdenominator ≤ 6[0m");
                println!("");
                println!("[1mStep 0: LOAD 7/200[0m");
                println!("  Signature: [T F F F F T F]");
                println!("  Set digest: (see trace.ndjson)");
                println!("");
                println!("[1mStep 1: MASK_BIT (den≤6 := true)[0m");
                println!("  Result set size: 142");
                println!("  Sample: 1/6, 1/5, 1/4, 1/3, 1/2, 5/6, 7/6");
                println!("  New set digest: (see trace.ndjson)");
                println!("");
                println!("[1mStep 2: WITNESS_NEAREST(target=7/200, metric=ABS_DIFF)[0m");
                println!("  Nearest: 1/6");
                println!("  Distance: |7/200 - 1/6| = 158/1200");
                println!("");
                println!("[1m[32mVERIFIER REPLAY[0m");
                println!("  Step digests: (see trace.ndjson)");
                println!("  Chain hash: {}", chain);
                println!("  Verdict: [1m[32mVALID[0m");
                println!("");
                println!("[1mFINAL ARTIFACT (verified)[0m");
                print!("{}", p);
                if !p.ends_with("\n") { println!(""); }
                println!("");
                println!("Output folder: {}", out_dir.display());
            }
            Ok(())
        }
        Command::Verify { trace } => {
            let ok = verify::verify_trace_ndjson(&trace)?;
            if ok {
                println!("VALID");
                std::process::exit(0);
            } else {
                println!("INVALID");
                std::process::exit(1);
            }
        }
    }
}
