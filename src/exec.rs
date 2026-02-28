use crate::digest::{merkle_root, sha256_bytes};
use crate::qe::{build_qe, canonical_cmp, parse_frac, Frac};
use crate::semtrace::{bit_legend, sig7, Constraint, Op, Trace};
use anyhow::{anyhow, Result};
use serde::Serialize;
use std::fs;
use std::path::PathBuf;
use time::OffsetDateTime;

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
    args: serde_json::Value,
    pre: StepPre,
    post: StepPost,
    step_digest: String,
}

#[derive(Clone, Debug)]
struct State {
    qe: Vec<Frac>,
    cst: Constraint,
    set: Vec<Frac>, // sorted canonical
    set_digest: [u8; 32],
    witness: Option<Frac>,
}

fn canonical_set_digest(set: &[Frac]) -> [u8; 32] {
    let mut leaves: Vec<[u8; 32]> = Vec::with_capacity(set.len());
    for f in set {
        leaves.push(sha256_bytes(&f.canonical_bytes()));
    }
    merkle_root(&leaves)
}

fn hex32(b: [u8; 32]) -> String {
    hex::encode(b)
}

fn step_digest(pre: &[u8], op: &str, args: &serde_json::Value, post: &[u8]) -> [u8; 32] {
    let obj = serde_json::json!({
        "pre": hex::encode(pre),
        "op": op,
        "args": args,
        "post": hex::encode(post),
    });
    // canonicalization note: serde_json preserves map insertion order in construction here.
    // For v0 this is sufficient; v1 should use a canonical JSON encoder identical across langs.
    let bytes = serde_json::to_vec(&obj).expect("json encode");
    sha256_bytes(&bytes)
}

fn filter_qe(qe: &[Frac], cst: Constraint) -> Vec<Frac> {
    let mut out = Vec::new();
    for f in qe {
        let s = sig7(f);
        if cst.matches(s) {
            out.push(*f);
        }
    }
    out.sort_by(canonical_cmp);
    out
}

fn frac_to_string(f: &Frac) -> String {
    format!("{}/{}", f.num, f.den)
}

fn distance_num_den(target: &Frac, cand: &Frac) -> (i64, i64) {
    // |a/b - c/d| = |ad - bc| / (bd)
    let a = target.num as i64;
    let b = target.den as i64;
    let c = cand.num as i64;
    let d = cand.den as i64;
    ((a * d - b * c).abs(), b * d)
}

fn dist_lt(x: (i64, i64), y: (i64, i64)) -> bool {
    // compare x.num/x.den < y.num/y.den by cross multiply
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

pub fn run_trace_and_write(trace: &Trace, out: Option<PathBuf>) -> Result<PathBuf> {
    if trace.universe != "QE" || trace.bits != 7 {
        return Err(anyhow!("unsupported universe/bits for v0"));
    }
    let qe = build_qe();

    // initial empty state
    let mut state = State {
        qe,
        cst: Constraint::empty(),
        set: Vec::new(),
        set_digest: sha256_bytes(b""),
        witness: None,
    };

    let out_dir = match out {
        Some(p) => p,
        None => {
            let now = OffsetDateTime::now_utc();
            let stamp = now
                .format(
                    &time::format_description::parse("[year][month][day]_[hour][minute][second]Z")
                        .unwrap(),
                )
                .unwrap();
            PathBuf::from("runs").join(stamp)
        }
    };
    fs::create_dir_all(&out_dir)?;
    // paragraph.txt is generated from the executed trace (human-readable narrowing path)
    let legend = bit_legend();
    let mut narrative: Vec<String> = Vec::new();
    narrative.push(format!(
        "REASONING TRACE (semtrace v{})",
        trace.semtrace_version
    ));
    narrative.push(format!(
        "Universe: {} (bits={})",
        trace.universe, trace.bits
    ));
    narrative.push(String::new());
    narrative.push("Bit legend:".to_string());
    for (i, name) in legend.iter().enumerate() {
        narrative.push(format!("  {}: {}", i, name));
    }
    narrative.push(String::new());
    narrative.push("Steps:".to_string());

    // Compute minimal constraint basis (QE-relative): drop redundant SET_BIT ops that do not change the filtered set.
    // NOTE: we compute this BEFORE execution so trace.ndjson + chain hash correspond to executed ops only.
    let mut exec_ops: Vec<Op> = Vec::new();
    let mut skipped: Vec<String> = Vec::new();
    {
        let mut cst = Constraint::empty();
        let mut cur_set: Vec<Frac> = Vec::new(); // will become QE after START_ELEM
        for op in trace.ops.iter() {
            match op {
                Op::StartElem { .. } => {
                    cst = Constraint::empty();
                    cur_set = state.qe.clone();
                    exec_ops.push(op.clone());
                }
                Op::SetBit { i, b } => {
                    let next_cst = cst.set_bit(*i, *b);
                    let next_set = filter_qe(&state.qe, next_cst);
                    if !cur_set.is_empty() && next_set == cur_set {
                        let bit_name = legend.get(*i as usize).copied().unwrap_or("?");
                        skipped.push(format!(
                            "SKIP_REDUNDANT SET_BIT {}:{}={}  (no change; count stays {})",
                            i,
                            bit_name,
                            b,
                            cur_set.len()
                        ));
                        continue;
                    }
                    cst = next_cst;
                    cur_set = next_set;
                    exec_ops.push(op.clone());
                }
                _ => {
                    exec_ops.push(op.clone());
                }
            }
        }
    }

    let mut ndjson = String::new();
    let mut chain: [u8; 32] = sha256_bytes(b""); // seed

    for (k, op) in exec_ops.iter().enumerate() {
        let pre = StepPre {
            set_digest: if state.set.is_empty() {
                None
            } else {
                Some(hex32(state.set_digest))
            },
            count: state.set.len(),
            constraint_mask: state.cst.mask,
            constraint_value: state.cst.value,
        };
        let (op_name, args_json, post_state_digest): (String, serde_json::Value, [u8; 32]) =
            match op {
                Op::StartElem { elem } => {
                    let f = parse_frac(elem).ok_or_else(|| anyhow!("bad frac: {}", elem))?;
                    // v0 semantics: START_ELEM grounds the target but does not constrain the set.
                    // Constraints are applied by subsequent ops (e.g. SET_BIT).
                    state.cst = Constraint::empty();
                    state.set = state.qe.clone();
                    state.set_digest = canonical_set_digest(&state.set);
                    state.witness = Some(f);
                    (
                        "START_ELEM".to_string(),
                        serde_json::json!({"elem": elem}),
                        state.set_digest,
                    )
                }
                Op::SetBit { i, b } => {
                    state.cst = state.cst.set_bit(*i, *b);
                    state.set = filter_qe(&state.qe, state.cst);
                    if state.set.is_empty() {
                        return Err(anyhow!("ERROR_EMPTY_SET"));
                    }
                    state.set_digest = canonical_set_digest(&state.set);
                    (
                        "SET_BIT".to_string(),
                        serde_json::json!({"i": i, "b": b}),
                        state.set_digest,
                    )
                }
                Op::WitnessNearest {
                    target_elem,
                    metric,
                } => {
                    if metric != "ABS_DIFF" {
                        return Err(anyhow!("unsupported metric"));
                    }
                    let target =
                        parse_frac(target_elem).ok_or_else(|| anyhow!("bad target frac"))?;
                    let w = witness_nearest(&state.set, &target)
                        .ok_or_else(|| anyhow!("ERROR_EMPTY_SET"))?;
                    state.witness = Some(w);
                    // witness op doesn't change set; use set_digest as post digest
                    (
                        "WITNESS_NEAREST".to_string(),
                        serde_json::json!({"target_elem": target_elem, "metric": metric}),
                        state.set_digest,
                    )
                }
                Op::ReturnSet {
                    max_items,
                    include_witness,
                } => {
                    let _ = (max_items, include_witness);
                    (
                        "RETURN_SET".to_string(),
                        serde_json::json!({"max_items": max_items, "include_witness": include_witness}),
                        state.set_digest,
                    )
                }
            };
        let post = StepPost {
            set_digest: Some(hex32(state.set_digest)),
            count: state.set.len(),
            witness: state.witness.map(|f| frac_to_string(&f)),
        };
        // narrative line (what changed)
        match op {
            Op::StartElem { elem } => {
                narrative.push(format!(
                    "  {}. START_ELEM elem={}  (set := QE; count {} → {})",
                    k, elem, pre.count, post.count
                ));
            }
            Op::SetBit { i, b } => {
                let bit_name = legend.get(*i as usize).copied().unwrap_or("?");
                let removed = pre.count.saturating_sub(post.count);
                let pct = if pre.count == 0 {
                    0.0
                } else {
                    (removed as f64) * 100.0 / (pre.count as f64)
                };
                narrative.push(format!(
                    "  {}. SET_BIT {}:{}={}  (removed {} = {:.2}%; count {} → {})",
                    k, i, bit_name, b, removed, pct, pre.count, post.count
                ));
            }
            Op::WitnessNearest {
                target_elem,
                metric,
            } => {
                narrative.push(format!(
                    "  {}. WITNESS_NEAREST target={} metric={}  (witness := {})",
                    k,
                    target_elem,
                    metric,
                    post.witness.clone().unwrap_or_else(|| "?".to_string())
                ));
            }
            Op::ReturnSet {
                max_items,
                include_witness,
            } => {
                narrative.push(format!("  {}. RETURN_SET max_items={} include_witness={}  (final count={}; chain continues)", k, max_items, include_witness, post.count));
            }
        }

        let sd = step_digest(&chain, &op_name, &args_json, &post_state_digest);
        chain = sd;

        let rec = StepRec {
            step: k,
            op: op_name,
            args: args_json,
            pre,
            post,
            step_digest: hex32(sd),
        };
        ndjson.push_str(&serde_json::to_string(&rec)?);
        ndjson.push('\n');
    }

    if !skipped.is_empty() {
        narrative.push(String::new());
        narrative.push("Redundant constraints removed:".to_string());
        for s in skipped {
            narrative.push(format!("  - {}", s));
        }
    }

    fs::write(out_dir.join("trace.ndjson"), &ndjson)?;
    fs::write(out_dir.join("paragraph.txt"), narrative.join("\n"))?;

    // result.json
    #[derive(Serialize)]
    struct ResultOut {
        verdict: String,
        chain_hash: String,
        count: usize,
        witness: Option<String>,
        bit_legend: Vec<String>,
        constraint: serde_json::Value,
        sample: Vec<String>,
    }
    let sample: Vec<String> = state.set.iter().take(20).map(frac_to_string).collect();
    let result = ResultOut {
        verdict: "OK".to_string(),
        chain_hash: hex32(chain),
        count: state.set.len(),
        witness: state.witness.map(|f| frac_to_string(&f)),
        bit_legend: bit_legend().iter().map(|s| s.to_string()).collect(),
        constraint: serde_json::json!({"mask": state.cst.mask, "value": state.cst.value}),
        sample,
    };
    fs::write(
        out_dir.join("result.json"),
        serde_json::to_string_pretty(&result)?,
    )?;

    // proof.json (v0: replayable digest proof)
    #[derive(Serialize)]
    struct Proof {
        verdict: String,
        chain_hash: String,
        note: String,
    }
    let proof = Proof {
        verdict: "VALID_IF_REPLAY_MATCHES".to_string(),
        chain_hash: hex32(chain),
        note: "v0 proof is determinism + replay: verifier recomputes each step digest and must match trace.ndjson".to_string(),
    };
    fs::write(
        out_dir.join("proof.json"),
        serde_json::to_string_pretty(&proof)?,
    )?;

    // digests.json
    #[derive(Serialize)]
    struct Digests {
        domain: serde_json::Value,
        tests: serde_json::Value,
        chain_hash: String,
    }
    // domain digest: hash of merkle root of full QE set
    let qe_digest = canonical_set_digest(&state.qe);
    let domain = serde_json::json!({"qe_merkle_root": hex32(qe_digest), "qe_size": state.qe.len()});
    // tests hash: hash of predicate legend bytes (placeholder v0)
    let tests = serde_json::json!({"predicate_legend": bit_legend(), "bits": 7});
    let dig = Digests {
        domain,
        tests,
        chain_hash: hex32(chain),
    };
    fs::write(
        out_dir.join("digests.json"),
        serde_json::to_string_pretty(&dig)?,
    )?;

    Ok(out_dir)
}
