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
            println!("{}", out_dir.display());
            Ok(())
        }
        Command::Exec { trace, out } => {
            let t = semtrace::read_trace_json(&trace)
                .with_context(|| format!("read trace json: {}", trace.display()))?;
            let out_dir = exec::run_trace_and_write(&t, out)?;
            println!("{}", out_dir.display());
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
