use anyhow::{anyhow, Result};
use std::path::PathBuf;

mod digest;
mod qe;
mod geom;
mod semtrace;
mod exec;
mod verify;
mod gpt2;

use semtrace::{Op, Trace};

fn parse_kv_u8(s: &str, key: &str) -> Option<u8> {
    // matches "... key=123 ..." where separator is '='
    for part in s.split_whitespace() {
        if let Some(rest) = part.strip_prefix(key) {
            if let Some(v) = rest.strip_prefix('=') {
                return v.parse::<u8>().ok();
            }
        }
    }
    None
}

fn proposer_ops_to_trace(ops: &[String]) -> Result<Trace> {
    let mut out: Vec<Op> = Vec::new();

    for op in ops {
        if let Some(elem) = op.strip_prefix("LOAD ") {
            out.push(Op::StartElem { elem: elem.trim().to_string() });
            continue;
        }

        if op.starts_with("MASK_BIT") {
            let i = parse_kv_u8(op, "bit").ok_or_else(|| anyhow!("bad MASK_BIT (missing bit=): {}", op))?;
            let b = parse_kv_u8(op, "val").ok_or_else(|| anyhow!("bad MASK_BIT (missing val=): {}", op))?;
            out.push(Op::SetBit { i, b });
            continue;
        }

        if let Some(target) = op.strip_prefix("WITNESS_NEAREST target=") {
            out.push(Op::WitnessNearest {
                target_elem: target.trim().to_string(),
                metric: "ABS_DIFF".to_string(),
            });
            continue;
        }

        if op.trim() == "RETURN_SET" {
            out.push(Op::ReturnSet { max_items: 20, include_witness: true });
            continue;
        }

        return Err(anyhow!("unknown proposer op: {}", op));
    }

    Ok(Trace {
        semtrace_version: "0.0.1".to_string(),
        universe: "QE".to_string(),
        bits: 7,
        ops: out,
    })
}

fn main() -> Result<()> {
    let query = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "Find fractions similar to 7/200 but with denominator ≤ 6".to_string());

    println!("");
    println!("\u{1b}[1mQUERY:\u{1b}[0m {}", query);
    println!("");

    let trace: Trace = if query.trim_start().starts_with("{") {
        serde_json::from_str::<Trace>(&query)
            .map_err(|e| anyhow!("bad trace json: {}", e))?
    } else {
        let proposer = gpt2::GPT2Proposer::new()?;
        let trace_ops = proposer.generate_trace(&query)?;

        println!("\u{1b}[1mPROPOSER OPS (raw):\u{1b}[0m");
        for op in &trace_ops {
            println!("  {}", op);
        }
        println!("");

        proposer_ops_to_trace(&trace_ops)?
    };

    // Execute (writes trace.ndjson, result.json, proof.json, paragraph.txt)
    let out_dir: PathBuf = exec::run_trace_and_write(&trace, None)?;

    // Verify (replay trace.ndjson and check digests/state)
    let trace_path = out_dir.join("trace.ndjson");
    let ok = verify::verify_trace_ndjson(&trace_path)?;

    println!("\u{1b}[1mEXECUTOR OUTPUT:\u{1b}[0m {}", out_dir.display());
    println!("  wrote: {}", trace_path.display());
    println!("  wrote: {}", out_dir.join("result.json").display());
    println!("  wrote: {}", out_dir.join("proof.json").display());
    println!("  wrote: {}", out_dir.join("paragraph.txt").display());
    println!("");

    if ok {
        println!("\u{1b}[1m\u{1b}[32mVERIFIER:\u{1b}[0m \u{1b}[1m\u{1b}[32mVALID (replay matched)\u{1b}[0m");
    } else {
        println!("\u{1b}[1m\u{1b}[31mVERIFIER:\u{1b}[0m \u{1b}[1m\u{1b}[31mINVALID (replay mismatch)\u{1b}[0m");
        return Err(anyhow!("verify failed"));
    }

    Ok(())
}
