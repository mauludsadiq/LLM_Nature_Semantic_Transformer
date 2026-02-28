use anyhow::{Error, Result};
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
    for part in s.split_whitespace() {
        if let Some(rest) = part.strip_prefix(key) {
            if let Some(v) = rest.strip_prefix('=') {
                return v.parse::<u8>().ok();
            }
        }
    }
    None
}

fn parse_kv_usize(s: &str, key: &str) -> Option<usize> {
    for part in s.split_whitespace() {
        if let Some(rest) = part.strip_prefix(key) {
            if let Some(v) = rest.strip_prefix('=') {
                return v.parse::<usize>().ok();
            }
        }
    }
    None
}

fn parse_kv_bool(s: &str, key: &str) -> Option<bool> {
    for part in s.split_whitespace() {
        if let Some(rest) = part.strip_prefix(key) {
            if let Some(v0) = rest.strip_prefix('=') {
                let v = v0.trim().trim_matches('\'').trim_matches('"');
                if v.eq_ignore_ascii_case("true") {
                    return Some(true);
                }
                if v.eq_ignore_ascii_case("false") {
                    return Some(false);
                }
            }
        }
    }
    None
}

fn proposer_ops_to_trace(ops: &[String]) -> Result<Trace> {
    let mut out: Vec<Op> = Vec::new();
    let mut universe = "QE".to_string();

    for op in ops {
        // QE v0: LOAD <frac>
        if let Some(elem) = op.strip_prefix("LOAD ") {
            out.push(Op::StartElem { elem: elem.trim().to_string() });
            continue;
        }

        // GE v0: START_ELEM <a,b,c>
        if let Some(elem) = op.strip_prefix("START_ELEM ") {
            universe = "GE".to_string();
            out.push(Op::StartElem { elem: elem.trim().to_string() });
            continue;
        }

        // QE v0: MASK_BIT bit=<i> val=<0/1>
        if op.starts_with("MASK_BIT") {
            let i = parse_kv_u8(op, "bit")
                .ok_or_else(|| Error::msg(format!("bad MASK_BIT (missing bit=): {}", op)))?;
            let b = parse_kv_u8(op, "val")
                .ok_or_else(|| Error::msg(format!("bad MASK_BIT (missing val=): {}", op)))?;
            out.push(Op::SetBit { i, b });
            continue;
        }

        // GE v0: SET_BIT i=<i> b=<0/1>
        if op.starts_with("SET_BIT") {
            universe = "GE".to_string();
            let i = parse_kv_u8(op, "i")
                .ok_or_else(|| Error::msg(format!("bad SET_BIT (missing i=): {}", op)))?;
            let b = parse_kv_u8(op, "b")
                .ok_or_else(|| Error::msg(format!("bad SET_BIT (missing b=): {}", op)))?;
            out.push(Op::SetBit { i, b });
            continue;
        }

        // QE v0: WITNESS_NEAREST target=<frac>
        if let Some(target) = op.strip_prefix("WITNESS_NEAREST target=") {
            out.push(Op::WitnessNearest {
                target_elem: target.trim().to_string(),
                metric: "ABS_DIFF".to_string(),
            });
            continue;
        }

        // GE v0: WITNESS_NEAREST target_elem=<a,b,c> metric=ABS_DIFF
        if op.starts_with("WITNESS_NEAREST") {
            if let Some(rest) = op.strip_prefix("WITNESS_NEAREST target_elem=") {
                universe = "GE".to_string();
                let target_elem = rest.split_whitespace().next().unwrap_or("").trim().to_string();
                let metric = op
                    .split_whitespace()
                    .find_map(|p| p.strip_prefix("metric=").map(|v| v.trim().to_string()))
                    .unwrap_or_else(|| "ABS_DIFF".to_string());
                out.push(Op::WitnessNearest { target_elem, metric });
                continue;
            }
        }

        // QE v0: RETURN_SET
        // GE v0: RETURN_SET max_items=<n> include_witness=<bool>
        if op.trim() == "RETURN_SET" || op.starts_with("RETURN_SET ") {
            let max_items = parse_kv_usize(op, "max_items").unwrap_or(20);
            let include_witness = parse_kv_bool(op, "include_witness").unwrap_or(true);
            out.push(Op::ReturnSet { max_items, include_witness });
            continue;
        }

        return Err(Error::msg(format!("unknown proposer op: {}", op)));
    }

    Ok(Trace {
        semtrace_version: "0.0.1".to_string(),
        universe,
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
            .map_err(|e| Error::msg(format!("bad trace json: {}", e)))?
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
        return Err(Error::msg("verify failed"));
    }

    Ok(())
}
