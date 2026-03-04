use anyhow::{anyhow, Result};
use serde::Serialize;
use serde_json::{json, Value as JsonValue};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[allow(unused_imports)]
use crate::word::{build_word_universe, canonical_cmp as word_canonical_cmp, is_word_universe, sig_distance, Word};
use crate::syllable::{build_syllable_universe, is_syllable_universe, sig_distance as syllable_sig_distance, Syllable};
use crate::morpheme::{build_morpheme_universe, is_morpheme_universe, sig_distance as morpheme_sig_distance, Morpheme};
use crate::phrase::{build_phrase_inventory, is_phrase_universe, sig_distance as phrase_sig_distance, Phrase};
use crate::semantic::{build_semantic_inventory, is_semantic_universe, sig_distance as semantic_sig_distance, SemanticGraph};
use crate::discourse::{build_discourse_inventory, is_discourse_universe, sig_distance as discourse_sig_distance, DiscourseGraph};
use crate::boolfun::{
    build_boolfun, canonical_cmp as boolfun_canonical_cmp, is_boolfun_universe,
    parse_elem as parse_boolfun, BoolFun,
};
use crate::digest::{merkle_root, sha256_bytes};
use crate::qe::{build_qe, canonical_cmp, parse_frac, Frac};
use crate::semtrace::{sig7, sig7_geom, Constraint};

#[derive(Debug)]
pub struct ExecutionResult {
    pub valid: bool,
    pub final_count: usize,
    pub witness: Option<String>,
    pub artifacts_path: Option<PathBuf>,
    pub universe: String,
    pub constraint_mask: u8,
    pub constraint_value: u8,
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
                universe = Some(t.trim_start_matches("universe=").trim_end_matches(|c: char| c == ';' || c == ',').to_string());
                continue;
            }
            if n.is_none() {
                n = parse_kv_u64(t, "n");
                if n.is_some() {
                    continue;
                }
            }
            if universe.is_none() && j == 1 && !t.contains("=") {
                universe = Some(t.trim_end_matches(|c: char| c == ';' || c == ',').to_string());
            }
        }
        let universe = universe.ok_or_else(|| anyhow!("SELECT_UNIVERSE missing universe="))?;
        let n = n.unwrap_or(0) as u8;
        return Ok((
            "SELECT_UNIVERSE".to_string(),
            json!({ "universe": universe, "n": n }),
        ));
    }

    if s.starts_with("FILTER_WEIGHT") {
        // expected: FILTER_WEIGHT min=1 max=3
        let toks: Vec<&str> = s.split_whitespace().collect();
        let mut min: Option<u64> = None;
        let mut max: Option<u64> = None;
        for t in toks.iter().skip(1) {
            if min.is_none() {
                min = parse_kv_u64(t, "min");
            }
            if max.is_none() {
                max = parse_kv_u64(t, "max");
            }
        }
        let min = min.ok_or_else(|| anyhow!("FILTER_WEIGHT missing min="))? as u32;
        let max = max.ok_or_else(|| anyhow!("FILTER_WEIGHT missing max="))? as u32;
        return Ok((
            "FILTER_WEIGHT".to_string(),
            json!({ "min": min, "max": max }),
        ));
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
            if k.is_none() {
                k = parse_kv_u64(t, "k");
            }
        }
        let target_elem = target.ok_or_else(|| anyhow!("TOPK missing target="))?;
        let k = k.ok_or_else(|| anyhow!("TOPK missing k="))? as usize;
        return Ok((
            "TOPK".to_string(),
            json!({ "target_elem": target_elem, "k": k }),
        ));
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
                metric = Some(
                    t.trim_start_matches("metric=")
                        .trim_end_matches(|c: char| c == ';' || c == ',')
                        .to_string(),
                );
            }
        }
        let target_elem = target.ok_or_else(|| anyhow!("WITNESS_NEAREST missing target="))?;
        let metric = metric.unwrap_or_else(|| "ABS_DIFF".to_string());
        return Ok((
            "WITNESS_NEAREST".to_string(),
            json!({ "target_elem": target_elem, "metric": metric }),
        ));
    }

    if s.starts_with("JOIN_NEAREST") {
        // expected: JOIN_NEAREST left_universe=QE right_universe=BOOLFUN left_elem=7/200 right_elem=0xBEEF metric=ABS_DIFF
        let toks: Vec<&str> = s.split_whitespace().collect();
        let mut left_universe: Option<String> = None;
        let mut right_universe: Option<String> = None;
        let mut left_elem: Option<String> = None;
        let mut right_elem: Option<String> = None;
        let mut metric: Option<String> = None;

        for t in toks.iter().skip(1) {
            if left_universe.is_none() && t.starts_with("left_universe=") {
                left_universe = Some(t.trim_start_matches("left_universe=").to_string());
                continue;
            }
            if right_universe.is_none() && t.starts_with("right_universe=") {
                right_universe = Some(t.trim_start_matches("right_universe=").to_string());
                continue;
            }
            if left_elem.is_none() && t.starts_with("left_elem=") {
                left_elem = Some(t.trim_start_matches("left_elem=").to_string());
                continue;
            }
            if right_elem.is_none() && t.starts_with("right_elem=") {
                right_elem = Some(t.trim_start_matches("right_elem=").to_string());
                continue;
            }
            if metric.is_none() && t.starts_with("metric=") {
                metric = Some(
                    t.trim_start_matches("metric=")
                        .trim_end_matches(|c: char| c == ';' || c == ',')
                        .to_string(),
                );
                continue;
            }
        }

        let left_universe =
            left_universe.ok_or_else(|| anyhow!("JOIN_NEAREST missing left_universe="))?;
        let right_universe =
            right_universe.ok_or_else(|| anyhow!("JOIN_NEAREST missing right_universe="))?;
        let left_elem = left_elem.ok_or_else(|| anyhow!("JOIN_NEAREST missing left_elem="))?;
        let right_elem = right_elem.ok_or_else(|| anyhow!("JOIN_NEAREST missing right_elem="))?;
        let metric = metric.unwrap_or_else(|| "ABS_DIFF".to_string());

        return Ok((
            "JOIN_NEAREST".to_string(),
            json!({
                "left_universe": left_universe,
                "right_universe": right_universe,
                "left_elem": left_elem,
                "right_elem": right_elem,
                "metric": metric
            }),
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
    if s.starts_with("PROJECT_SIGNATURE") {
        let toks: Vec<&str> = s.split_whitespace().collect();
        let elem = toks
            .iter()
            .skip(1)
            .find_map(|t| t.strip_prefix("elem="))
            .or_else(|| toks.get(1).copied())
            .ok_or_else(|| anyhow!("PROJECT_SIGNATURE missing elem"))?;
        return Ok(("PROJECT_SIGNATURE".to_string(), json!({ "elem": elem })));
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
    let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S%.6fZ").to_string();
    let thread_id = format!("{:?}", std::thread::current().id()).replace("ThreadId(", "").replace(")", "");
    let artifacts_dir = PathBuf::from("runs").join(format!("{}_{}", timestamp, thread_id));
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
    let mut active_universe: String = "QE".to_string();
    let mut set_digest: [u8; 32] = sha256_bytes(b"");
    let mut witness: Option<Frac> = None;
    let mut witness_bf: Option<BoolFun> = None;
    let mut is_ge: bool = false;

    let mut word_all: Vec<Word> = Vec::new();
    let mut word_set: Vec<Word> = Vec::new();
    let mut is_word: bool = false;
    let mut witness_word: Option<Word> = None;
    let _ = &witness_word; // read via is_word branches below
    let mut syllable_all: Vec<Syllable> = Vec::new();
    let mut syllable_set: Vec<Syllable> = Vec::new();
    let mut witness_syllable: Option<Syllable> = None;
    let _ = &witness_syllable;
    let mut morpheme_all: Vec<Morpheme> = Vec::new();
    let mut morpheme_set: Vec<Morpheme> = Vec::new();
    let mut witness_morpheme: Option<Morpheme> = None;
    let _ = &witness_morpheme;
    let mut phrase_all: Vec<Phrase> = Vec::new();
    let mut phrase_set: Vec<Phrase> = Vec::new();
    let mut witness_phrase: Option<Phrase> = None;
    let _ = &witness_phrase;
    let mut semantic_all: Vec<SemanticGraph> = Vec::new();
    let mut semantic_set: Vec<SemanticGraph> = Vec::new();
    let mut witness_semantic: Option<SemanticGraph> = None;
    let _ = &witness_semantic;
    let mut discourse_all: Vec<DiscourseGraph> = Vec::new();
    let mut discourse_set: Vec<DiscourseGraph> = Vec::new();
    let mut witness_discourse: Option<DiscourseGraph> = None;
    let _ = &witness_discourse;
    let mut is_syllable = false;
    let mut is_morpheme = false;
    let mut is_phrase = false;
    let mut is_semantic = false;
    let mut is_discourse = false;

    let mut chain: [u8; 32] = sha256_bytes(b"");

    // RETURN_SET params for result output
    let mut want_max_items: usize = 20;
    let mut want_include_witness: bool = false;

    let mut out_lines: Vec<String> = Vec::with_capacity(ops.len());

    for (step_idx, raw_op) in ops.iter().enumerate() {
        let (op, args) = parse_op_to_semtrace(raw_op)?;

        let pre = StepPre {
            set_digest: if step_idx == 0
                && ((is_boolfun && boolfun_set.is_empty()) || (!is_boolfun && state_set.is_empty()))
            {
                None
            } else {
                Some(hex32(set_digest))
            },
            count: if is_boolfun {
                boolfun_set.len()
            } else {
                state_set.len()
            },
            constraint_mask: cst.mask,
            constraint_value: cst.value,
        };

        match op.as_str() {
            "SELECT_UNIVERSE" => {
                let u = args
                    .get("universe")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow!("bad args for SELECT_UNIVERSE"))?;
                let n = args.get("n").and_then(|v| v.as_u64()).unwrap_or(0) as u8;

                let u_norm = u.to_ascii_uppercase();
                active_universe = u_norm.clone();

                // BOOLFUN
                if is_boolfun_universe(u_norm.as_str()) {
                    is_boolfun = true;
                    is_ge = false;
                    cst = Constraint::empty();
                    state_set.clear();
                    boolfun_n = n;
                    boolfun_all = build_boolfun(n);
                    boolfun_set = boolfun_all.clone();
                    boolfun_set.sort_by(boolfun_canonical_cmp);
                    set_digest = canonical_set_digest_boolfun(&boolfun_set);
                    witness = None;
                    witness_bf = None;
                } else if u_norm == "QE" {
                    // QE (fractions)
                    is_boolfun = false;
                    is_ge = false;
                    cst = Constraint::empty();
                    state_set = qe.clone();
                    set_digest = canonical_set_digest(&state_set);
                    witness = None;
                    witness_bf = None;
                } else if is_word_universe(u_norm.as_str()) {
                    is_boolfun = false;
                    is_ge = false;
                    is_word = true;
                    cst = Constraint::empty();
                    state_set.clear();
                    if word_all.is_empty() {
                        word_all = build_word_universe();
                    }
                    word_set = word_all.clone();
                    set_digest = {
                        let leaves: Vec<[u8; 32]> = word_set.iter()
                            .map(|w| sha256_bytes(&w.canonical_bytes()))
                            .collect();
                        crate::digest::merkle_root(&leaves)
                    };
                    witness = None;
                    witness_bf = None;
                    witness_word = None;
                } else if is_syllable_universe(u_norm.as_str()) {
                    is_boolfun=false; is_ge=false; is_word=false; is_syllable=true;
                    is_morpheme=false; is_phrase=false; is_semantic=false; is_discourse=false;
                    cst=Constraint::empty(); state_set.clear();
                    if syllable_all.is_empty() { syllable_all=build_syllable_universe(); }
                    syllable_set=syllable_all.clone();
                    set_digest={let mut l:Vec<[u8;32]>=syllable_set.iter().map(|s|sha256_bytes(&s.canonical_bytes())).collect();l.sort_unstable();merkle_root(&l)};
                    witness=None; witness_bf=None; witness_syllable=None;
                } else if is_morpheme_universe(u_norm.as_str()) {
                    is_boolfun=false; is_ge=false; is_word=false; is_syllable=false;
                    is_morpheme=true; is_phrase=false; is_semantic=false; is_discourse=false;
                    cst=Constraint::empty(); state_set.clear();
                    if morpheme_all.is_empty() { morpheme_all=build_morpheme_universe(); }
                    morpheme_set=morpheme_all.clone();
                    set_digest={let mut l:Vec<[u8;32]>=morpheme_set.iter().map(|m|sha256_bytes(&m.canonical_bytes())).collect();l.sort_unstable();merkle_root(&l)};
                    witness=None; witness_bf=None; witness_morpheme=None;
                } else if is_phrase_universe(u_norm.as_str()) {
                    is_boolfun=false; is_ge=false; is_word=false; is_syllable=false;
                    is_morpheme=false; is_phrase=true; is_semantic=false; is_discourse=false;
                    cst=Constraint::empty(); state_set.clear();
                    if phrase_all.is_empty() { phrase_all=build_phrase_inventory(); }
                    phrase_set=phrase_all.clone();
                    set_digest={let mut l:Vec<[u8;32]>=phrase_set.iter().map(|p|sha256_bytes(&p.canonical_bytes())).collect();l.sort_unstable();merkle_root(&l)};
                    witness=None; witness_bf=None; witness_phrase=None;
                } else if is_semantic_universe(u_norm.as_str()) {
                    is_boolfun=false; is_ge=false; is_word=false; is_syllable=false;
                    is_morpheme=false; is_phrase=false; is_semantic=true; is_discourse=false;
                    cst=Constraint::empty(); state_set.clear();
                    if semantic_all.is_empty() { semantic_all=build_semantic_inventory(); }
                    semantic_set=semantic_all.clone();
                    set_digest={let mut l:Vec<[u8;32]>=semantic_set.iter().map(|g|sha256_bytes(&g.canonical_bytes())).collect();l.sort_unstable();merkle_root(&l)};
                    witness=None; witness_bf=None; witness_semantic=None;
                } else if is_discourse_universe(u_norm.as_str()) {
                    is_boolfun=false; is_ge=false; is_word=false; is_syllable=false;
                    is_morpheme=false; is_phrase=false; is_semantic=false; is_discourse=true;
                    cst=Constraint::empty(); state_set.clear();
                    if discourse_all.is_empty() { discourse_all=build_discourse_inventory(); }
                    discourse_set=discourse_all.clone();
                    set_digest={let mut l:Vec<[u8;32]>=discourse_set.iter().map(|g|sha256_bytes(&g.canonical_bytes())).collect();l.sort_unstable();merkle_root(&l)};
                    witness=None; witness_bf=None; witness_discourse=None;
                } else {
                    return Err(anyhow!("unsupported universe: {}", u));
                }
            }
            "FILTER_WEIGHT" => {
                if !is_boolfun {
                    return Err(anyhow!("FILTER_WEIGHT requires BOOLFUN universe"));
                }
                let min = args
                    .get("min")
                    .and_then(|v| v.as_u64())
                    .ok_or_else(|| anyhow!("bad args for FILTER_WEIGHT"))?
                    as u32;
                let max = args
                    .get("max")
                    .and_then(|v| v.as_u64())
                    .ok_or_else(|| anyhow!("bad args for FILTER_WEIGHT"))?
                    as u32;
                let mut out: Vec<BoolFun> = boolfun_all
                    .iter()
                    .copied()
                    .filter(|f| {
                        let w = f.weight();
                        w >= min && w <= max
                    })
                    .collect();
                out.sort_by(boolfun_canonical_cmp);
                boolfun_set = out;
                set_digest = canonical_set_digest_boolfun(&boolfun_set);
            }
            "TOPK" => {
                if !is_boolfun {
                    return Err(anyhow!("TOPK requires BOOLFUN universe"));
                }
                let target_s = args
                    .get("target_elem")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow!("bad args for TOPK"))?;
                let k = args
                    .get("k")
                    .and_then(|v| v.as_u64())
                    .ok_or_else(|| anyhow!("bad args for TOPK"))? as usize;
                let target =
                    parse_boolfun(target_s).ok_or_else(|| anyhow!("bad boolfun target"))?;
                if boolfun_n == 0 {
                    boolfun_n = target.n;
                    boolfun_all = build_boolfun(boolfun_n);
                    boolfun_set = boolfun_all.clone();
                    boolfun_set.sort_by(boolfun_canonical_cmp);
                }
                if target.n != boolfun_n {
                    return Err(anyhow!(
                        "boolfun target n mismatch: have={} want={}",
                        target.n,
                        boolfun_n
                    ));
                }

                let mut scored: Vec<(u32, BoolFun)> = boolfun_set
                    .iter()
                    .copied()
                    .map(|f| (f.hamming(&target), f))
                    .collect();
                scored.sort_by(|(da, fa), (db, fb)| {
                    da.cmp(db).then_with(|| boolfun_canonical_cmp(fa, fb))
                });
                let take = k.min(scored.len());
                let top: Vec<BoolFun> = scored.into_iter().take(take).map(|(_, f)| f).collect();
                boolfun_set = top;
                boolfun_set.sort_by(boolfun_canonical_cmp);
                set_digest = canonical_set_digest_boolfun(&boolfun_set);
                witness_bf = boolfun_set.get(0).copied();
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
                    let mut v: Vec<Frac> = tris
                        .into_iter()
                        .map(|t| Frac { num: t.a, den: t.c })
                        .collect();
                    v.sort_by(crate::qe::canonical_cmp);
                    state_set = v;
                    set_digest = canonical_set_digest(&state_set);
                    is_boolfun = false;
                    witness_bf = None;
                    witness = Some(Frac { num: a, den: c });
                } else {
                    let f = parse_frac(elem).ok_or_else(|| anyhow!("bad frac elem"))?;
                    state_set = qe.clone();
                    set_digest = canonical_set_digest(&state_set);
                    is_boolfun = false;
                    witness_bf = None;
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
                    let mut v: Vec<Frac> = tris
                        .into_iter()
                        .map(|t| Frac { num: t.a, den: t.c })
                        .collect();
                    v.sort_by(crate::qe::canonical_cmp);
                    state_set = v;
                } else {
                    state_set = filter_qe(&qe, cst);
                    set_digest = canonical_set_digest(&state_set);
                }
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
                if is_syllable && metric == "HAMMING_SIG" {
                    let t_idx: usize = target.trim().parse().unwrap_or(0);
                    if let Some(ts) = syllable_all.get(t_idx).cloned() {
                        witness_syllable = syllable_set.iter().min_by_key(|s| syllable_sig_distance(s, &ts)).cloned();
                    }
                } else if is_word && metric == "HAMMING_SIG" {
                    // Word universe: nearest by signature Hamming distance
                    let t_text = target.trim().to_ascii_lowercase();
                    let t_word = word_all.iter().find(|w| w.text == t_text)
                        .cloned()
                        .or_else(|| crate::word::parse_elem(&t_text))
                        .ok_or_else(|| anyhow!("word not found: {}", target))?;
                    let best = word_set.iter()
                        .min_by_key(|w| sig_distance(w, &t_word))
                        .cloned()
                        .ok_or_else(|| anyhow!("empty word set"))?;
                    witness_word = Some(best);
                } else if is_morpheme && metric == "HAMMING_SIG" {
                    let t_norm = target.trim().to_ascii_lowercase();
                    if let Some(tm) = morpheme_all.iter().find(|m| m.meaning_id.ends_with(&t_norm)).cloned() {
                        witness_morpheme = morpheme_set.iter().min_by_key(|m| morpheme_sig_distance(m, &tm)).cloned();
                    }
                } else if is_phrase && metric == "HAMMING_SIG" {
                    let t_id: u32 = target.trim().parse().unwrap_or(1);
                    if let Some(tp) = phrase_all.iter().find(|p| p.phrase_id == t_id).cloned() {
                        witness_phrase = phrase_set.iter().min_by_key(|p| phrase_sig_distance(p, &tp)).cloned();
                    }
                } else if is_semantic && metric == "HAMMING_SIG" {
                    let t_id: u32 = target.trim().parse().unwrap_or(1);
                    if let Some(tg) = semantic_all.iter().find(|g| g.graph_id == t_id).cloned() {
                        witness_semantic = semantic_set.iter().min_by_key(|g| semantic_sig_distance(g, &tg)).cloned();
                    }
                } else if is_discourse && metric == "HAMMING_SIG" {
                    let t_id: u32 = target.trim().parse().unwrap_or(1);
                    if let Some(tg) = discourse_all.iter().find(|g| g.discourse_id == t_id).cloned() {
                        witness_discourse = discourse_set.iter().min_by_key(|g| discourse_sig_distance(g, &tg)).cloned();
                    }
                } else if metric == "ABS_DIFF" {
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
                } else {
                    return Err(anyhow!("unsupported metric: {}", metric));
                }
            }
            "PROJECT_SIGNATURE" => {
                let elem = args
                    .get("elem")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow!("bad args for PROJECT_SIGNATURE"))?;
                let f = parse_frac(elem).ok_or_else(|| anyhow!("bad frac elem"))?;
                let sig: u64 = (crate::semtrace::sig7(&f) as u64) & 0x7f;

                // QE -> 7-bit signature -> BOOLFUN signature universe (n=7, bits in 0..127)
                is_boolfun = true;
                is_ge = false;
                cst = Constraint::empty();
                state_set.clear();
                boolfun_n = 7;
                boolfun_all = build_boolfun(7);
                boolfun_set = boolfun_all.clone();
                boolfun_set.sort_by(boolfun_canonical_cmp);
                set_digest = canonical_set_digest_boolfun(&boolfun_set);
                witness_bf = Some(BoolFun { n: 7, bits: sig });
            }
            "JOIN_NEAREST" => {
                let metric = args
                    .get("metric")
                    .and_then(|v| v.as_str())
                    .unwrap_or("ABS_DIFF");
                if metric != "ABS_DIFF" && metric != "HAMMING" {
                    return Err(anyhow!("unsupported join metric: {}", metric));
                }

                let lu = args
                    .get("left_universe")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow!("JOIN_NEAREST missing left_universe"))?;
                let ru = args
                    .get("right_universe")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow!("JOIN_NEAREST missing right_universe"))?;
                let le = args
                    .get("left_elem")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow!("JOIN_NEAREST missing left_elem"))?;
                let re = args
                    .get("right_elem")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow!("JOIN_NEAREST missing right_elem"))?;

                let lu_norm = lu.to_ascii_uppercase();
                let ru_norm = ru.to_ascii_uppercase();

                if lu_norm == "QE" && is_boolfun_universe(ru_norm.as_str()) {
                    let bf = parse_boolfun(re).ok_or_else(|| anyhow!("bad right_elem"))?;
                    witness_bf = Some(bf);
                    cst.mask = 0x7f;
                    cst.value = (bf.bits as u8) & 0x7f;

                    state_set = filter_qe(&qe, cst);
                    set_digest = canonical_set_digest(&state_set);

                    let t = parse_frac(le).ok_or_else(|| anyhow!("bad left_elem"))?;
                    let w = witness_nearest(&state_set, &t).ok_or_else(|| anyhow!("empty set"))?;
                    witness = Some(w);
                } else {
                    return Err(anyhow!(
                        "JOIN_NEAREST unsupported join: left_universe={} right_universe={}",
                        lu,
                        ru
                    ));
                }
            }

            "RETURN_SET" => {
                want_max_items =
                    args.get("max_items").and_then(|v| v.as_u64()).unwrap_or(20) as usize;
                want_include_witness = args
                    .get("include_witness")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
            }
            _ => return Err(anyhow!("unknown semtrace op: {}", op)),
        }

        let post = StepPost {
            set_digest: Some(hex32(set_digest)),
            count: if is_boolfun {
                boolfun_set.len()
            } else {
                state_set.len()
            },
            witness: if is_boolfun {
                witness_bf.as_ref().map(boolfun_to_string)
            } else {
                witness.as_ref().map(frac_to_string)
            },
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

    let witness_s = if is_boolfun {
        witness_bf.as_ref().map(boolfun_to_string)
    } else if is_syllable {
        witness_syllable.as_ref().map(|s| format!("syllable:{}", String::from_utf8_lossy(&s.canonical_bytes()).chars().take(40).collect::<String>()))
    } else if is_word {
        witness_word.as_ref().map(|w| w.text.clone())
    } else if is_morpheme {
        witness_morpheme.as_ref().map(|m| m.meaning_id.to_string())
    } else if is_phrase {
        witness_phrase.as_ref().map(|p| format!("phrase:{}", p.phrase_id))
    } else if is_semantic {
        witness_semantic.as_ref().map(|g| format!("graph:{}", g.graph_id))
    } else if is_discourse {
        witness_discourse.as_ref().map(|g| format!("discourse:{}", g.discourse_id))
    } else {
        witness.as_ref().map(frac_to_string)
    };

    let mut sample: Vec<String> = Vec::new();

    if want_include_witness {
        if let Some(w) = witness_s.as_ref() {
            sample.push(w.clone());
        }
    }

    let remain = want_max_items.saturating_sub(sample.len());

    if is_boolfun {
        let mut pushed = 0usize;
        for f in boolfun_set.iter() {
            if pushed >= remain {
                break;
            }
            if let Some(w) = witness_bf.as_ref() {
                if *f == *w {
                    continue;
                }
            }
            sample.push(boolfun_to_string(f));
            pushed += 1;
        }
    } else {
        let mut pushed = 0usize;
        for f in state_set.iter() {
            if pushed >= remain {
                break;
            }
            if let Some(w) = witness.as_ref() {
                if *f == *w {
                    continue;
                }
            }
            sample.push(frac_to_string(f));
            pushed += 1;
        }
    }

    let set_nonempty = if is_boolfun {
        !boolfun_set.is_empty()
    } else {
        !state_set.is_empty()
    };
    let verdict_ok = replay_ok;
    let result = json!({
        "verdict": if set_nonempty { "OK" } else { "EMPTY_SET" },
        "verifier": { "valid": replay_ok },
        "chain_hash": hex32(chain),
        "count": if is_boolfun { boolfun_set.len() } else if is_word { word_set.len() } else if is_syllable { syllable_set.len() } else if is_morpheme { morpheme_set.len() } else if is_phrase { phrase_set.len() } else if is_semantic { semantic_set.len() } else if is_discourse { discourse_set.len() } else { state_set.len() },
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
        witness
            .as_ref()
            .map(frac_to_string)
            .unwrap_or_else(|| "(none)".to_string()),
    );
    fs::write(&paragraph_path, paragraph)?;

    let elapsed = start.elapsed();
    if verbose {
        println!("⏱️  Execution completed in {:.2?}", elapsed);
        println!("📁 Artifacts written to: {}", artifacts_dir.display());
    }

    Ok(ExecutionResult {
        valid: verdict_ok,
        final_count: if is_boolfun { boolfun_set.len()
        } else if is_word { word_set.len()
        } else if is_syllable { syllable_set.len()
        } else if is_morpheme { morpheme_set.len()
        } else if is_phrase { phrase_set.len()
        } else if is_semantic { semantic_set.len()
        } else if is_discourse { discourse_set.len()
        } else { state_set.len() },
        witness: witness_s,
        artifacts_path: Some(artifacts_dir),
        universe: active_universe.clone(),
        constraint_mask: cst.mask,
        constraint_value: cst.value,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn project_signature_roundtrip() {
        let ops = vec![
            "PROJECT_SIGNATURE elem=7/200".to_string(),
            "RETURN_SET max_items=8 include_witness=1".to_string(),
        ];
        let result = run_trace_and_write(&ops, None, false).unwrap();
        // Must be valid (exec and verify agree)
        assert!(result.valid, "verifier must agree with executor");
        // Universe defaults to QE (no SELECT_UNIVERSE op)
        assert_eq!(result.universe, "QE");
        // sig7(7/200): positive=1, rat_int=0, den<=6=0, num_even=0, den_mod3=0, proper=1, num_abs<=5=0
        // = 0b0100001 = 33
        assert_eq!(result.witness.as_deref(), Some("u64:33"),
            "sig7(7/200) should be 33");
    }

    #[test]
    fn project_signature_integer_elem() {
        let ops = vec![
            "PROJECT_SIGNATURE elem=3/1".to_string(),
            "RETURN_SET max_items=4 include_witness=1".to_string(),
        ];
        let result = run_trace_and_write(&ops, None, false).unwrap();
        assert!(result.valid);
        // sig7(3/1) = 0b1000111 = 71 (from semtrace tests)
        assert_eq!(result.witness.as_deref(), Some("u64:71"),
            "sig7(3/1) should be 71");
    }
    #[test]
    fn word_universe_witness_nearest() {
        let ops = vec![
            "SELECT_UNIVERSE universe=WORD n=0".to_string(),
            "WITNESS_NEAREST target_elem=abandon metric=HAMMING_SIG".to_string(),
            "RETURN_SET max_items=5 include_witness=1".to_string(),
        ];
        let result = run_trace_and_write(&ops, None, false).unwrap();
        assert!(result.valid, "execution must verify");
        assert!(result.witness.is_some(), "must have a witness");
        // abandon is in the universe so distance=0, witness=abandon itself
        assert_eq!(result.witness.as_deref(), Some("abandon"));
    }

}
