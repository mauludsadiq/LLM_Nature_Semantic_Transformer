use anyhow::{anyhow, Result};
use serde::Serialize;
use serde_json::{json, Value as JsonValue};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::digest::{merkle_root, sha256_bytes};
use crate::qe::{build_qe, canonical_cmp, parse_frac, Frac};
use crate::semtrace::{sig7, sig7_geom, Constraint};
#[allow(unused_imports)]
use crate::boolfun::{build_boolfun, parse_elem as parse_boolfun, canonical_cmp as boolfun_canonical_cmp, BoolFun};

#[derive(Debug)]
pub struct ExecutionResult {
    pub valid: bool,
    pub final_count: usize,
    pub witness: Option<String>,
    pub artifacts_path: Option<PathBuf>,
}

#[derive(Clone, Debug, Serialize)]
struct StepPre {
    set_digest: Option<String>,
    count: usize,
    constraint_mask: u8,
    constraint_value: u8,
}

#[derive(Clone, Debug, Serialize)]
struct StepPost {
    set_digest: Option<String>,
    count: usize,
    witness: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
struct StepRec {
    step: usize,
    op: String,
    args: JsonValue,
    pre: StepPre,
    post: StepPost,
    step_digest: String,
}

fn hex32(b: [u8; 32]) -> String {
    hex::encode(b)
}

fn canonical_set_digest(set: &[Frac]) -> [u8; 32] {
    let mut leaves: Vec<[u8; 32]> = Vec::with_capacity(set.len());
    for f in set {
        leaves.push(sha256_bytes(&f.canonical_bytes()));
    }
    merkle_root(&leaves)
}

fn canonical_set_digest_boolfun(set: &[BoolFun]) -> [u8; 32] {
    let mut leaves: Vec<[u8; 32]> = Vec::with_capacity(set.len());
    for f in set {
        leaves.push(sha256_bytes(&f.canonical_bytes()));
    }
    merkle_root(&leaves)
}

fn step_digest(pre_chain: &[u8], op: &str, args: &JsonValue, post_set: &[u8]) -> [u8; 32] {
    let obj = json!({
        "pre": hex::encode(pre_chain),
        "op": op,
        "args": args,
        "post": hex::encode(post_set),
    });
    let bytes = serde_json::to_vec(&obj).expect("json encode");
    sha256_bytes(&bytes)
}

fn frac_to_string(f: &Frac) -> String {
    format!("{}/{}", f.num, f.den)
}

fn boolfun_to_string(f: &BoolFun) -> String {
    if f.n == 4 {
        format!("0x{:04X}", (f.bits & 0xFFFF) as u16)
    } else {
        format!("u64:{}", f.bits)
    }
}

fn distance_num_den(target: &Frac, cand: &Frac) -> (i64, i64) {
    let a = target.num as i64;
    let b = target.den as i64;
    let c = cand.num as i64;
    let d = cand.den as i64;
    ((a * d - b * c).abs(), b * d)
}

fn dist_lt(x: (i64, i64), y: (i64, i64)) -> bool {
    x.0 * y.1 < y.0 * x.1
}

fn witness_nearest(set: &[Frac], target: &Frac) -> Option<Frac> {
    if set.is_empty() {
        return None;
    }
    let mut best = set[0];
    let mut best_d = distance_num_den(target, &best);
    for f in set.iter().skip(1) {
        let d = distance_num_den(target, f);
        let better = dist_lt(d, best_d)
            || (d == best_d && (f.num.abs(), f.den) < (best.num.abs(), best.den))
            || (d == best_d
                && (f.num.abs(), f.den) == (best.num.abs(), best.den)
                && canonical_cmp(f, &best).is_lt());
        if better {
            best = *f;
            best_d = d;
        }
    }
    Some(best)
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

fn parse_kv_u64(tok: &str, key: &str) -> Option<u64> {
    let prefix = format!("{key}=");
    if !tok.starts_with(&prefix) {
        return None;
    }
    tok[prefix.len()..].parse().ok()
}

fn parse_kv_bool(tok: &str, key: &str) -> Option<bool> {
    let prefix = format!("{key}=");
    if !tok.starts_with(&prefix) {
        return None;
    }
    let v = &tok[prefix.len()..];
    match v {
        "1" | "true" | "TRUE" | "True" => Some(true),
        "0" | "false" | "FALSE" | "False" => Some(false),
        _ => None,
    }
}

fn parse_op_to_semtrace(op: &str) -> Result<(String, JsonValue)> {
    let s = op.trim();

    if let Some(rest) = s.strip_prefix("LOAD ") {
        let elem = rest.trim();
        if elem.is_empty() {
            return Err(anyhow!("LOAD missing elem"));
        }
        return Ok(("START_ELEM".to_string(), json!({ "elem": elem })));
    }

    if s.starts_with("MASK_BIT") {
        // expected: MASK_BIT bit=2 val=1
        let toks: Vec<&str> = s.split_whitespace().collect();
        let mut bit: Option<u64> = None;
        let mut val: Option<u64> = None;
        for t in toks.iter().skip(1) {
            if bit.is_none() {
                bit = parse_kv_u64(t, "bit");
            }
            if val.is_none() {
                val = parse_kv_u64(t, "val");
            }
        }
        let i = bit.ok_or_else(|| anyhow!("MASK_BIT missing bit="))? as u8;
        let b = val.ok_or_else(|| anyhow!("MASK_BIT missing val="))? as u8;
        return Ok(("SET_BIT".to_string(), json!({ "i": i, "b": b })));
    }

    if s.starts_with("SELECT_UNIVERSE") {
        // expected: SELECT_UNIVERSE universe=BOOLFUN n=4  (or: SELECT_UNIVERSE BoolFun n=4)
        let toks: Vec<&str> = s.split_whitespace().collect();
        let mut universe: Option<String> = None;
        let mut n: Option<u64> = None;
        for (j, t) in toks.iter().enumerate().skip(1) {
            if universe.is_none() && t.starts_with("universe=") {
                universe = Some(t.trim_start_matches("universe=").to_string());
                continue;
            }
            if n.is_none() {
                n = parse_kv_u64(t, "n");
                if n.is_some() { continue; }
            }
            if universe.is_none() && j == 1 && !t.contains("=") {
                universe = Some(t.to_string());
            }
        }
        let universe = universe.ok_or_else(|| anyhow!("SELECT_UNIVERSE missing universe="))?;
        let n = n.ok_or_else(|| anyhow!("SELECT_UNIVERSE missing n="))? as u8;
        return Ok(("SELECT_UNIVERSE".to_string(), json!({ "universe": universe, "n": n })));
    }

    if s.starts_with("FILTER_WEIGHT") {
        // expected: FILTER_WEIGHT min=1 max=3
        let toks: Vec<&str> = s.split_whitespace().collect();
        let mut min: Option<u64> = None;
        let mut max: Option<u64> = None;
        for t in toks.iter().skip(1) {
            if min.is_none() { min = parse_kv_u64(t, "min"); }
            if max.is_none() { max = parse_kv_u64(t, "max"); }
        }
        let min = min.ok_or_else(|| anyhow!("FILTER_WEIGHT missing min="))? as u32;
        let max = max.ok_or_else(|| anyhow!("FILTER_WEIGHT missing max="))? as u32;
        return Ok(("FILTER_WEIGHT".to_string(), json!({ "min": min, "max": max })));
    }

    if s.starts_with("TOPK") {
        // expected: TOPK target=0xBEEF k=5
        let toks: Vec<&str> = s.split_whitespace().collect();
        let mut target: Option<String> = None;
        let mut k: Option<u64> = None;
        for t in toks.iter().skip(1) {
            if target.is_none() && t.starts_with("target=") {
                target = Some(t.trim_start_matches("target=").to_string());
            }
            if target.is_none() && t.starts_with("target_elem=") {
                target = Some(t.trim_start_matches("target_elem=").to_string());
            }
            if k.is_none() { k = parse_kv_u64(t, "k"); }
        }
        let target_elem = target.ok_or_else(|| anyhow!("TOPK missing target="))?;
        let k = k.ok_or_else(|| anyhow!("TOPK missing k="))? as usize;
        return Ok(("TOPK".to_string(), json!({ "target_elem": target_elem, "k": k })));
    }


    if s.starts_with("WITNESS_NEAREST") {
        // expected: WITNESS_NEAREST target=13/37 (metric defaults ABS_DIFF)
        let toks: Vec<&str> = s.split_whitespace().collect();
        let mut target: Option<String> = None;
        let mut metric: Option<String> = None;
        for t in toks.iter().skip(1) {
            if target.is_none() && t.starts_with("target=") {
                target = Some(t.trim_start_matches("target=").to_string());
            }
            if target.is_none() && t.starts_with("target_elem=") {
                target = Some(t.trim_start_matches("target_elem=").to_string());
            }
            if metric.is_none() && t.starts_with("metric=") {
                metric = Some(t.trim_start_matches("metric=").to_string());
            }
        }
        let target_elem = target.ok_or_else(|| anyhow!("WITNESS_NEAREST missing target="))?;
        let metric = metric.unwrap_or_else(|| "ABS_DIFF".to_string());
        return Ok((
            "WITNESS_NEAREST".to_string(),
            json!({ "target_elem": target_elem, "metric": metric }),
        ));
    }

    if s.starts_with("RETURN_SET") {
        // expected: RETURN_SET max_items=10 include_witness=true
        let toks: Vec<&str> = s.split_whitespace().collect();
        let mut max_items: usize = 20;
        let mut include_witness: bool = false;
        for t in toks.iter().skip(1) {
            if let Some(v) = parse_kv_u64(t, "max_items") {
                max_items = v as usize;
            }
            if let Some(v) = parse_kv_bool(t, "include_witness") {
                include_witness = v;
            }
        }
        return Ok((
            "RETURN_SET".to_string(),
            json!({ "max_items": max_items, "include_witness": include_witness }),
        ));
    }

    Err(anyhow!("unknown op: {}", s))
}

pub fn run_trace_and_write(
    ops: &[String],
    _trace_path: Option<&Path>,
    verbose: bool,
) -> Result<ExecutionResult> {
    let start = Instant::now();

    // Artifacts dir
    let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%SZ").to_string();
    let artifacts_dir = PathBuf::from("runs").join(timestamp);
    fs::create_dir_all(&artifacts_dir)?;

    let trace_ndjson_path = artifacts_dir.join("trace.ndjson");
    let proof_path = artifacts_dir.join("proof.json");
    let result_path = artifacts_dir.join("result.json");
    let paragraph_path = artifacts_dir.join("paragraph.txt");

    // Universe state
    let qe = build_qe();
    let ge_state = crate::geom::build_ge(20);

    let mut boolfun_all: Vec<BoolFun> = Vec::new();
    let mut boolfun_set: Vec<BoolFun> = Vec::new();
    let mut boolfun_n: u8 = 0;
    let mut is_boolfun: bool = false;

    let mut state_set: Vec<Frac> = Vec::new();
    let mut cst = Constraint::empty();
    let mut set_digest: [u8; 32] = sha256_bytes(b"");
    let mut witness: Option<Frac> = None;
    let mut witness_bf: Option<BoolFun> = None;
    let mut is_ge: bool = false;

    let mut chain: [u8; 32] = sha256_bytes(b"");

    // RETURN_SET params for result output
    let mut want_max_items: usize = 20;
    let mut want_include_witness: bool = false;

    let mut out_lines: Vec<String> = Vec::with_capacity(ops.len());

    for (step_idx, raw_op) in ops.iter().enumerate() {
        let (op, args) = parse_op_to_semtrace(raw_op)?;

        let pre = StepPre {
            set_digest: if step_idx == 0 && ((is_boolfun && boolfun_set.is_empty()) || (!is_boolfun && state_set.is_empty())) {
                None
            } else {
                Some(hex32(set_digest))
            },
            count: if is_boolfun { boolfun_set.len() } else { state_set.len() },
            constraint_mask: cst.mask,
            constraint_value: cst.value,
        };

        match op.as_str() {
            "SELECT_UNIVERSE" => {
                let u = args.get("universe").and_then(|v| v.as_str()).ok_or_else(|| anyhow!("bad args for SELECT_UNIVERSE"))?;
                let n = args.get("n").and_then(|v| v.as_u64()).ok_or_else(|| anyhow!("bad args for SELECT_UNIVERSE"))? as u8;

                let u_norm = u.to_ascii_uppercase();
                if u_norm == "BOOLFUN" || u_norm == "BOOLFUN<N>" || u_norm == "BOOLFUN4" || u_norm == "BOOLFUN_4" || u_norm == "BOOLFUNV0" || u_norm == "BOOLFUNV1" || u_norm == "BOOLFUNS" || u_norm == "BOOLFUNS<N>" || u_norm == "BOOLFUNS4" || u_norm == "BOOLFUNS_4" || u_norm == "BOOLFUNS_V0" || u_norm == "BOOLFUNS_V1" || u_norm == "BOOLFUNS" || u_norm == "BOOLFUNS<N>" || u_norm == "BOOLFUNS4" || u_norm == "BOOLFUNS_4" || u_norm == "BOOLFUNS_V0" || u_norm == "BOOLFUNS_V1" || u_norm == "BOOLFUNS" || u_norm == "BOOLFUNS<N>" || u_norm == "BOOLFUNS4" || u_norm == "BOOLFUNS_4" || u_norm == "BOOLFUNS_V0" || u_norm == "BOOLFUNS_V1" || u_norm == "BOOLFUNS" || u_norm == "BOOLFUNS<N>" || u_norm == "BOOLFUNS4" || u_norm == "BOOLFUNS_4" || u_norm == "BOOLFUNS_V0" || u_norm == "BOOLFUNS_V1" || u_norm == "BOOLFUNS" || u_norm == "BOOLFUNS<N>" || u_norm == "BOOLFUNS4" || u_norm == "BOOLFUNS_4" || u_norm == "BOOLFUNS_V0" || u_norm == "BOOLFUNS_V1" || u_norm == "BOOLFUNS" || u_norm == "BOOLFUNS<N>" || u_norm == "BOOLFUNS4" || u_norm == "BOOLFUNS_4" || u_norm == "BOOLFUNS_V0" || u_norm == "BOOLFUNS_V1" || u_norm == "BOOLFUNS" || u_norm == "BOOLFUNS<N>" || u_norm == "BOOLFUNS4" || u_norm == "BOOLFUNS_4" || u_norm == "BOOLFUNS_V0" || u_norm == "BOOLFUNS_V1" || u_norm == "BOOLFUNS" || u_norm == "BOOLFUNS<N>" || u_norm == "BOOLFUNS4" || u_norm == "BOOLFUNS_4" || u_norm == "BOOLFUNS_V0" || u_norm == "BOOLFUNS_V1" || u_norm == "BOOLFUNS" || u_norm == "BOOLFUNS<N>" || u_norm == "BOOLFUNS4" || u_norm == "BOOLFUNS_4" || u_norm == "BOOLFUNS_V0" || u_norm == "BOOLFUNS_V1" || u_norm == "BOOLFUNS" || u_norm == "BOOLFUNS<N>" || u_norm == "BOOLFUNS4" || u_norm == "BOOLFUNS_4" || u_norm == "BOOLFUNS_V0" || u_norm == "BOOLFUNS_V1" || u_norm == "BOOLFUNS" || u_norm == "BOOLFUNS<N>" || u_norm == "BOOLFUNS4" || u_norm == "BOOLFUNS_4" || u_norm == "BOOLFUNS_V0" || u_norm == "BOOLFUNS_V1" || u_norm == "BOOLFUNS" || u_norm == "BOOLFUNS<N>" || u_norm == "BOOLFUNS4" || u_norm == "BOOLFUNS_4" || u_norm == "BOOLFUNS_V0" || u_norm == "BOOLFUNS_V1" || u_norm == "BOOLFUNS" || u_norm == "BOOLFUNS<N>" || u_norm == "BOOLFUNS4" || u_norm == "BOOLFUNS_4" || u_norm == "BOOLFUNS_V0" || u_norm == "BOOLFUNS_V1" || u_norm == "BOOLFUNS" || u_norm == "BOOLFUNS<N>" || u_norm == "BOOLFUNS4" || u_norm == "BOOLFUNS_4" || u_norm == "BOOLFUNS_V0" || u_norm == "BOOLFUNS_V1" || u_norm == "BOOLFUN" || u_norm == "BOOLFUN<N>" || u_norm == "BOOLFUN4" || u_norm == "BOOLFUN_4" || u_norm == "BOOLFUNV0" || u_norm == "BOOLFUNV1" || u_norm == "BOOLFUNS" || u_norm == "BOOLFUNS<N>" || u_norm == "BOOLFUNS4" || u_norm == "BOOLFUNS_4" || u_norm == "BOOLFUNS_V0" || u_norm == "BOOLFUNS_V1" || u_norm == "BOOLFUN" || u_norm == "BOOLFUN<N>" {
                    // accept many spellings; canonical is BOOLFUN
                }

                is_boolfun = u_norm == "BOOLFUN" || u_norm == "BOOLFUN<N>" || u_norm == "BOOLFUN4" || u_norm == "BOOLFUN_4" || u_norm == "BOOLFUNV0" || u_norm == "BOOLFUNV1" || u_norm == "BOOLFUNS" || u_norm == "BOOLFUNS<N>" || u_norm == "BOOLFUNS4" || u_norm == "BOOLFUNS_4" || u_norm == "BOOLFUNS_V0" || u_norm == "BOOLFUNS_V1";
                if !is_boolfun {
                    return Err(anyhow!("unsupported universe: {}", u));
                }

                is_ge = false;
                cst = Constraint::empty();
                state_set.clear();
                witness = None;

                boolfun_n = n;
                boolfun_all = build_boolfun(n);
                boolfun_set = boolfun_all.clone();
                boolfun_set.sort_by(boolfun_canonical_cmp);
                set_digest = canonical_set_digest_boolfun(&boolfun_set);
                witness_bf = None;
            }
            "FILTER_WEIGHT" => {
                if !is_boolfun {
                    return Err(anyhow!("FILTER_WEIGHT requires BOOLFUN universe"));
                }
                let min = args.get("min").and_then(|v| v.as_u64()).ok_or_else(|| anyhow!("bad args for FILTER_WEIGHT"))? as u32;
                let max = args.get("max").and_then(|v| v.as_u64()).ok_or_else(|| anyhow!("bad args for FILTER_WEIGHT"))? as u32;
                let mut out: Vec<BoolFun> = boolfun_all.iter().copied().filter(|f| {
                    let w = f.weight();
                    w >= min && w <= max
                }).collect();
                out.sort_by(boolfun_canonical_cmp);
                boolfun_set = out;
                set_digest = canonical_set_digest_boolfun(&boolfun_set);
                witness_bf = None;
            }
            "TOPK" => {
                if !is_boolfun {
                    return Err(anyhow!("TOPK requires BOOLFUN universe"));
                }
                let target_s = args.get("target_elem").and_then(|v| v.as_str()).ok_or_else(|| anyhow!("bad args for TOPK"))?;
                let k = args.get("k").and_then(|v| v.as_u64()).ok_or_else(|| anyhow!("bad args for TOPK"))? as usize;
                let target = parse_boolfun(target_s).ok_or_else(|| anyhow!("bad boolfun target"))?;
                if target.n != boolfun_n {
                    return Err(anyhow!("boolfun target n mismatch: have={} want={}", target.n, boolfun_n));
                }

                let mut scored: Vec<(u32, BoolFun)> = boolfun_set.iter().copied().map(|f| (f.hamming(&target), f)).collect();
                scored.sort_by(|(da, fa), (db, fb)| {
                    da.cmp(db).then_with(|| boolfun_canonical_cmp(fa, fb))
                });
                let take = k.min(scored.len());
                let top: Vec<BoolFun> = scored.into_iter().take(take).map(|(_, f)| f).collect();
                witness_bf = top.get(0).copied();
                // state_set remains boolfun_set; digest unchanged
            }

            "START_ELEM" => {
                let elem = args
                    .get("elem")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow!("bad args for START_ELEM"))?;

                is_ge = elem.contains(',');

                cst = Constraint::empty();

                if is_ge {
                    let parts: Vec<&str> = elem
                        .split(',')
                        .map(|s| s.trim())
                        .filter(|s| !s.is_empty())
                        .collect();
                    if parts.len() != 3 {
                        return Err(anyhow!("bad tri elem"));
                    }
                    let a: i32 = parts[0].parse().map_err(|_| anyhow!("bad tri"))?;
                    let b: i32 = parts[1].parse().map_err(|_| anyhow!("bad tri"))?;
                    let c: i32 = parts[2].parse().map_err(|_| anyhow!("bad tri"))?;
                    crate::geom::Tri::new(a, b, c).ok_or_else(|| anyhow!("bad tri"))?;

                    let mut tris = ge_state.clone();
                    tris.sort_by(crate::geom::canonical_cmp);
                    let mut v: Vec<Frac> = tris.into_iter().map(|t| Frac { num: t.a, den: t.c }).collect();
                    v.sort_by(crate::qe::canonical_cmp);
                    state_set = v;

                    set_digest = canonical_set_digest(&state_set);
                    witness = Some(Frac { num: a, den: c });
                } else {
                    let f = parse_frac(elem).ok_or_else(|| anyhow!("bad frac elem"))?;
                    state_set = qe.clone();
                    set_digest = canonical_set_digest(&state_set);
                    witness = Some(f);
                }
            }
            "SET_BIT" => {
                let i = args
                    .get("i")
                    .and_then(|v| v.as_u64())
                    .ok_or_else(|| anyhow!("bad args for SET_BIT"))? as u8;
                let b = args
                    .get("b")
                    .and_then(|v| v.as_u64())
                    .ok_or_else(|| anyhow!("bad args for SET_BIT"))? as u8;

                cst = cst.set_bit(i, b);

                if is_ge {
                    let mut tris: Vec<crate::geom::Tri> = ge_state
                        .iter()
                        .copied()
                        .filter(|t| cst.matches(sig7_geom(t)))
                        .collect();
                    tris.sort_by(crate::geom::canonical_cmp);
                    let mut v: Vec<Frac> = tris.into_iter().map(|t| Frac { num: t.a, den: t.c }).collect();
                    v.sort_by(crate::qe::canonical_cmp);
                    state_set = v;
                } else {
                    state_set = filter_qe(&qe, cst);
                }

                set_digest = canonical_set_digest(&state_set);
            }
            "WITNESS_NEAREST" => {
                let target = args
                    .get("target_elem")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow!("bad args for WITNESS_NEAREST"))?;
                let metric = args
                    .get("metric")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow!("bad args for WITNESS_NEAREST"))?;
                if metric != "ABS_DIFF" {
                    return Err(anyhow!("unsupported metric: {}", metric));
                }

                let t: Frac = if is_ge || target.contains(',') {
                    let parts: Vec<&str> = target
                        .split(',')
                        .map(|s| s.trim())
                        .filter(|s| !s.is_empty())
                        .collect();
                    if parts.len() != 3 {
                        return Err(anyhow!("bad tri target"));
                    }
                    let a: i32 = parts[0].parse().map_err(|_| anyhow!("bad tri target"))?;
                    let b: i32 = parts[1].parse().map_err(|_| anyhow!("bad tri target"))?;
                    let c: i32 = parts[2].parse().map_err(|_| anyhow!("bad tri target"))?;
                    crate::geom::Tri::new(a, b, c).ok_or_else(|| anyhow!("bad tri target"))?;
                    Frac { num: a, den: c }
                } else {
                    parse_frac(target).ok_or_else(|| anyhow!("bad frac target"))?
                };

                let w = witness_nearest(&state_set, &t).ok_or_else(|| anyhow!("empty set"))?;
                witness = Some(w);
            }
            "RETURN_SET" => {
                want_max_items = args
                    .get("max_items")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(20) as usize;
                want_include_witness = args
                    .get("include_witness")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
            }
            _ => return Err(anyhow!("unknown semtrace op: {}", op)),
        }

        let post = StepPost {
            set_digest: Some(hex32(set_digest)),
            count: if is_boolfun { boolfun_set.len() } else { state_set.len() },
            witness: if is_boolfun { witness_bf.as_ref().map(boolfun_to_string) } else { witness.as_ref().map(frac_to_string) },
        };

        let sd = step_digest(&chain, &op, &args, &set_digest);
        chain = sd;

        let rec = StepRec {
            step: step_idx,
            op,
            args,
            pre,
            post,
            step_digest: hex32(sd),
        };

        out_lines.push(serde_json::to_string(&rec)?);
    }

    fs::write(&trace_ndjson_path, out_lines.join("\n") + "\n")?;

    let replay_ok = crate::verify::verify_trace_ndjson(&trace_ndjson_path)?;

    let proof = json!({
        "ops_in": ops,
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "trace_ndjson": trace_ndjson_path,
    });
    fs::write(&proof_path, serde_json::to_string_pretty(&proof)?)?;

    let witness_s = if is_boolfun { witness_bf.as_ref().map(boolfun_to_string) } else { witness.as_ref().map(frac_to_string) };

    let mut sample: Vec<String> = Vec::new();
      if is_boolfun {
          let n = want_max_items.min(boolfun_set.len());
          for f in boolfun_set.iter().take(n) {
              sample.push(boolfun_to_string(f));
          }
      } else {
          let n = want_max_items.min(state_set.len());
          for f in state_set.iter().take(n) {
              sample.push(frac_to_string(f));
          }
      }

    let set_nonempty = if is_boolfun { !boolfun_set.is_empty() } else { !state_set.is_empty() };
    let verdict_ok = replay_ok;
    let result = json!({
        "verdict": if set_nonempty { "OK" } else { "EMPTY_SET" },
        "verifier": { "valid": replay_ok },
        "chain_hash": hex32(chain),
        "count": if is_boolfun { boolfun_set.len() } else { state_set.len() },
        "witness": witness_s,
        "constraint": { "mask": cst.mask, "value": cst.value },
        "return_set": { "max_items": want_max_items, "include_witness": want_include_witness },
        "sample": sample,
        "artifacts": {
            "trace_ndjson": trace_ndjson_path,
            "proof": proof_path,
            "result": result_path,
            "paragraph": paragraph_path,
        }
    });
    fs::write(&result_path, serde_json::to_string_pretty(&result)?)?;

    let paragraph = format!(
        "Semantic Transformer (exec)\nchain_hash={}\ncount={}\nwitness={}\n",
        hex32(chain),
        state_set.len(),
        witness.as_ref().map(frac_to_string).unwrap_or_else(|| "(none)".to_string()),
    );
    fs::write(&paragraph_path, paragraph)?;

    let elapsed = start.elapsed();
    if verbose {
        println!("⏱️  Execution completed in {:.2?}", elapsed);
        println!("📁 Artifacts written to: {}", artifacts_dir.display());
    }

    Ok(ExecutionResult {
        valid: verdict_ok,
        final_count: if is_boolfun { boolfun_set.len() } else { state_set.len() },
        witness: witness_s,
        artifacts_path: Some(artifacts_dir),
    })
}

pub fn write_trace_to_file(ops: &[String], query: &str) -> Result<PathBuf> {
    let trace_dir = PathBuf::from("traces");
    fs::create_dir_all(&trace_dir)?;

    let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S");
    let trace_path = trace_dir.join(format!("trace_{}.json", timestamp));

    let trace = json!({
        "query": query,
        "ops": ops,
        "timestamp": chrono::Utc::now().to_rfc3339(),
    });

    fs::write(&trace_path, serde_json::to_string_pretty(&trace)?)?;

    Ok(trace_path)
}
