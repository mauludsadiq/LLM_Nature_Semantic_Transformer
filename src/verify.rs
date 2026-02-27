use crate::digest::{sha256_bytes, merkle_root};
use crate::qe::{build_qe, canonical_cmp, parse_frac, Frac};
use crate::semtrace::{sig7, Constraint};
use anyhow::{anyhow, Result};
use serde::Deserialize;
use std::fs;
use std::path::Path;

#[derive(Clone, Debug, Deserialize)]
#[allow(dead_code)]
struct StepPre {
    set_digest: Option<String>,
    count: usize,
    constraint_mask: u8,
    constraint_value: u8,
}

#[derive(Clone, Debug, Deserialize)]
struct StepPost {
    set_digest: Option<String>,
    count: usize,
    witness: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
#[allow(dead_code)]
struct StepRec {
    step: usize,
    op: String,
    args: serde_json::Value,
    pre: StepPre,
    post: StepPost,
    step_digest: String,
}

fn canonical_set_digest(set: &[Frac]) -> [u8; 32] {
    let mut leaves: Vec<[u8; 32]> = Vec::with_capacity(set.len());
    for f in set {
        leaves.push(sha256_bytes(&f.canonical_bytes()));
    }
    merkle_root(&leaves)
}

fn hex32(b: [u8; 32]) -> String { hex::encode(b) }

fn step_digest(pre: &[u8], op: &str, args: &serde_json::Value, post: &[u8]) -> [u8; 32] {
    let obj = serde_json::json!({
        "pre": hex::encode(pre),
        "op": op,
        "args": args,
        "post": hex::encode(post),
    });
    let bytes = serde_json::to_vec(&obj).expect("json encode");
    sha256_bytes(&bytes)
}

fn filter_qe(qe: &[Frac], cst: Constraint) -> Vec<Frac> {
    let mut out = Vec::new();
    for f in qe {
        if cst.matches(sig7(f)) {
            out.push(*f);
        }
    }
    out.sort_by(canonical_cmp);
    out
}

fn frac_to_string(f: &Frac) -> String { format!("{}/{}", f.num, f.den) }

fn distance_num_den(target: &Frac, cand: &Frac) -> (i64, i64) {
    let a = target.num as i64;
    let b = target.den as i64;
    let c = cand.num as i64;
    let d = cand.den as i64;
    ((a*d - b*c).abs(), b*d)
}
fn dist_lt(x: (i64,i64), y: (i64,i64)) -> bool { x.0 * y.1 < y.0 * x.1 }

fn witness_nearest(set: &[Frac], target: &Frac) -> Option<Frac> {
    if set.is_empty() { return None; }
    let mut best = set[0];
    let mut best_d = distance_num_den(target, &best);
    for f in set.iter().skip(1) {
        let d = distance_num_den(target, f);
        let better = dist_lt(d, best_d)
            || (d == best_d && (f.num.abs(), f.den) < (best.num.abs(), best.den))
            || (d == best_d && (f.num.abs(), f.den) == (best.num.abs(), best.den) && canonical_cmp(f, &best).is_lt());
        if better {
            best = *f;
            best_d = d;
        }
    }
    Some(best)
}

pub fn verify_trace_ndjson(trace_path: &Path) -> Result<bool> {
    let qe = build_qe();
    let txt = fs::read_to_string(trace_path)?;

    let mut state_set: Vec<Frac> = Vec::new();
    let mut cst = Constraint::empty();
    let mut set_digest = sha256_bytes(b"");
    let mut witness: Option<Frac> = None;

    let mut chain: [u8; 32] = sha256_bytes(b"");

    for line in txt.lines() {
        if line.trim().is_empty() { continue; }
        let rec: StepRec = serde_json::from_str(line)?;

        // recompute transition based on rec.op/args
        match rec.op.as_str() {
            "START_ELEM" => {
                let elem = rec.args.get("elem").and_then(|v| v.as_str()).ok_or_else(|| anyhow!("bad args"))?;
                let f = parse_frac(elem).ok_or_else(|| anyhow!("bad frac"))?;
                // v0 semantics: START_ELEM grounds the target but does not constrain the set.
                cst = Constraint::empty();
                state_set = qe.clone();
                set_digest = canonical_set_digest(&state_set);
                witness = Some(f);
            }
            "SET_BIT" => {
                let i = rec.args.get("i").and_then(|v| v.as_u64()).ok_or_else(|| anyhow!("bad args"))? as u8;
                let b = rec.args.get("b").and_then(|v| v.as_u64()).ok_or_else(|| anyhow!("bad args"))? as u8;
                cst = cst.set_bit(i, b);
                state_set = filter_qe(&qe, cst);
                if state_set.is_empty() { return Ok(false); }
                set_digest = canonical_set_digest(&state_set);
            }
            "WITNESS_NEAREST" => {
                let target = rec.args.get("target_elem").and_then(|v| v.as_str()).ok_or_else(|| anyhow!("bad args"))?;
                let metric = rec.args.get("metric").and_then(|v| v.as_str()).ok_or_else(|| anyhow!("bad args"))?;
                if metric != "ABS_DIFF" { return Ok(false); }
                let t = parse_frac(target).ok_or_else(|| anyhow!("bad target"))?;
                let w = witness_nearest(&state_set, &t).ok_or_else(|| anyhow!("empty"))?;
                witness = Some(w);
            }
            "RETURN_SET" => {
                // no-op for state
            }
            _ => return Ok(false),
        }

        // check post fields
        let post_set_hex = rec.post.set_digest.clone().unwrap_or_default();
        if post_set_hex != hex32(set_digest) { return Ok(false); }
        if rec.post.count != state_set.len() { return Ok(false); }
        if let Some(w) = witness {
            if rec.post.witness.as_deref() != Some(&frac_to_string(&w)) {
                // RETURN_SET might keep witness; still must match if present
                // if post witness omitted, ok; but our executor always includes if known
                return Ok(false);
            }
        }

        let sd = step_digest(&chain, &rec.op, &rec.args, &set_digest);
        chain = sd;
        if rec.step_digest != hex32(sd) { return Ok(false); }
    }

    Ok(true)
}
