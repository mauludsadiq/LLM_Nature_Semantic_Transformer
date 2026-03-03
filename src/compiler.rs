use anyhow::{anyhow, Result};
use crate::semtrace::{Op, Trace};

#[derive(Clone, Debug)]
pub struct Candidate {
    pub trace: Trace,
    pub score: f64,
    pub rationale: String,
}

fn contains_positive_hint(s: &str) -> bool {
    let q = s.to_ascii_lowercase();
    q.contains("positive") || q.contains("\u{2265}0") || q.contains(">=0") || q.contains("nonnegative")
}

fn contains_proper_hint(s: &str) -> bool {
    let q = s.to_ascii_lowercase();
    q.contains("proper") || q.contains("proper fraction")
}

fn contains_small_hint(s: &str) -> bool {
    let q = s.to_ascii_lowercase();
    q.contains("small") || q.contains("near") || q.contains("close") || q.contains("similar")
}

/// Deterministic multi-candidate compiler: natural-language-ish query -> candidate semtraces.
/// Score: lower is better. All candidates are fully auditable.
pub fn compile_query_to_candidates(query: &str) -> Result<Vec<Candidate>> {
    let q = query.trim();
    if q.is_empty() {
        return Err(anyhow!("empty query"));
    }

    // QE (fractions) candidates
    if let Some(fr) = first_frac(q) {
        let max_items = parse_max_items(q).unwrap_or(20);
        let include_witness = contains_include_witness(q);
        let want_den_le_6 = contains_den_le_6(q);
        let want_positive = contains_positive_hint(q);
        let want_proper = contains_proper_hint(q);
        let want_small = contains_small_hint(q);

        let mut base_ops: Vec<Op> = Vec::new();
        base_ops.push(Op::StartElem { elem: fr.clone() });
        if want_den_le_6 {
            base_ops.push(Op::SetBit { i: 2, b: 1 });
        }
        base_ops.push(Op::WitnessNearest { target_elem: fr.clone(), metric: "ABS_DIFF".to_string() });
        base_ops.push(Op::ReturnSet { max_items, include_witness });

        let base = Trace { semtrace_version: "0.0.1".to_string(), universe: "QE".to_string(), bits: 7, ops: base_ops.clone() };

        let mut out: Vec<Candidate> = Vec::new();

        // Candidate 0: literal (only explicit constraints)
        out.push(Candidate {
            trace: base.clone(),
            score: 0.0,
            rationale: "literal: only explicit constraints".to_string(),
        });

        // Candidate 1: +positive if hinted OR if user said similar/close (common intent)
        if want_positive || want_small {
            let mut ops = base_ops.clone();
            // Insert SET_BIT after START_ELEM (bit 0 = positive)
            ops.insert(1, Op::SetBit { i: 0, b: 1 });
            out.push(Candidate {
                trace: Trace { semtrace_version: "0.0.1".to_string(), universe: "QE".to_string(), bits: 7, ops },
                score: 0.10,
                rationale: "intent: prefer positive fractions (bit0=1)".to_string(),
            });
        }

        // Candidate 2: +proper if hinted OR if user said similar/close (common intent)
        if want_proper || want_small {
            let mut ops = base_ops.clone();
            ops.insert(1, Op::SetBit { i: 5, b: 1 }); // proper
            out.push(Candidate {
                trace: Trace { semtrace_version: "0.0.1".to_string(), universe: "QE".to_string(), bits: 7, ops },
                score: 0.15,
                rationale: "intent: prefer proper fractions (bit5=1)".to_string(),
            });
        }

        // Candidate 3: +small numerator magnitude if hinted OR if user said similar/close
        if want_small {
            let mut ops = base_ops.clone();
            ops.insert(1, Op::SetBit { i: 6, b: 1 }); // num_abs<=5
            out.push(Candidate {
                trace: Trace { semtrace_version: "0.0.1".to_string(), universe: "QE".to_string(), bits: 7, ops },
                score: 0.20,
                rationale: "intent: prefer small numerator magnitude (bit6=1)".to_string(),
            });
        }

        // Candidate 4: combined common intent: positive + proper (+ optional den<=6 already in base)
        if want_small {
            let mut ops = base_ops.clone();
            // insert in deterministic order after START_ELEM
            ops.insert(1, Op::SetBit { i: 0, b: 1 });
            ops.insert(2, Op::SetBit { i: 5, b: 1 });
            out.push(Candidate {
                trace: Trace { semtrace_version: "0.0.1".to_string(), universe: "QE".to_string(), bits: 7, ops },
                score: 0.25,
                rationale: "intent: prefer positive + proper (bit0=1, bit5=1)".to_string(),
            });
        }

        return Ok(out);
    }

    // BOOLFUN candidates (small subset)
    let ql = q.to_ascii_lowercase();
    if ql.contains("boolfun") {
        let mut n: Option<u8> = None;
        let mut target: Option<String> = None;
        let mut k: Option<usize> = None;

        if let Some(pos) = ql.find("n=") {
            let start = pos + 2;
            let mut end = start;
            let b = ql.as_bytes();
            while end < b.len() && b[end].is_ascii_digit() { end += 1; }
            if end > start {
                if let Ok(v) = ql[start..end].parse::<u8>() { n = Some(v); }
            }
        }

        if let Some(pos) = ql.find("target=") {
            let start = pos + "target=".len();
            let mut end = start;
            let b = q.as_bytes();
            while end < b.len() && !b[end].is_ascii_whitespace() { end += 1; }
            if end > start { target = Some(q[start..end].to_string()); }
        }

        if let Some(pos) = ql.find("k=") {
            let start = pos + 2;
            let mut end = start;
            let b = ql.as_bytes();
            while end < b.len() && b[end].is_ascii_digit() { end += 1; }
            if end > start {
                if let Ok(v) = ql[start..end].parse::<usize>() { k = Some(v); }
            }
        }

        if k.is_none() { k = parse_max_items(q); }

        let n = n.ok_or_else(|| anyhow!("BOOLFUN compile requires n=..."))?;
        let target = target.ok_or_else(|| anyhow!("BOOLFUN compile requires target=..."))?;
        let k = k.unwrap_or(10);

        let ops: Vec<Op> = vec![
            Op::SelectUniverse { universe: "BOOLFUN".to_string(), n },
            Op::TopK { target_elem: target, k },
            Op::ReturnSet { max_items: k, include_witness: true },
        ];

        return Ok(vec![Candidate {
            trace: Trace { semtrace_version: "0.0.1".to_string(), universe: "BOOLFUN".to_string(), bits: 7, ops },
            score: 0.0,
            rationale: "literal: BOOLFUN requires explicit n and target".to_string(),
        }]);
    }

    Err(anyhow!("unable to deterministically compile query; provide explicit JSON ops"))
}


fn first_frac(s: &str) -> Option<String> {
    // deterministic, no regex dependency: scan for digit+/digit+
    // Accepts patterns like "7/200" anywhere in the string.
    let bytes = s.as_bytes();
    for i in 0..bytes.len() {
        if !bytes[i].is_ascii_digit() {
            continue;
        }
        // consume numerator
        let mut j = i;
        while j < bytes.len() && bytes[j].is_ascii_digit() {
            j += 1;
        }
        if j >= bytes.len() || bytes[j] != b'/' {
            continue;
        }
        let slash = j;
        j += 1;
        if j >= bytes.len() || !bytes[j].is_ascii_digit() {
            continue;
        }
        while j < bytes.len() && bytes[j].is_ascii_digit() {
            j += 1;
        }
        let cand = &s[i..j];
        // must have at least one digit on each side
        if slash > i && j > slash + 1 {
            return Some(cand.to_string());
        }
    }
    None
}

fn contains_den_le_6(s: &str) -> bool {
    let q = s.to_ascii_lowercase();
    // cover common spellings: "den <= 6", "denominator <= 6", "den ≤ 6"
    q.contains("den<=6")
        || q.contains("den <= 6")
        || q.contains("den ≤ 6")
        || q.contains("denominator<=6")
        || q.contains("denominator <= 6")
        || q.contains("denominator ≤ 6")
        || q.contains("denom<=6")
        || q.contains("denom <= 6")
        || q.contains("denom ≤ 6")
}

fn contains_include_witness(s: &str) -> bool {
    let q = s.to_ascii_lowercase();
    q.contains("include_witness")
        || q.contains("include witness")
        || q.contains("show witness")
        || q.contains("with witness")
}

fn parse_max_items(s: &str) -> Option<usize> {
    // deterministic: look for "max_items=NN" or "max items NN" or "top NN"
    let q = s.to_ascii_lowercase();

    // max_items=NN
    if let Some(pos) = q.find("max_items=") {
        let start = pos + "max_items=".len();
        let mut end = start;
        let b = q.as_bytes();
        while end < b.len() && b[end].is_ascii_digit() {
            end += 1;
        }
        if end > start {
            if let Ok(v) = q[start..end].parse::<usize>() {
                return Some(v);
            }
        }
    }

    // "max items NN"
    if let Some(pos) = q.find("max items") {
        let mut k = pos + "max items".len();
        while k < q.len() && q.as_bytes()[k].is_ascii_whitespace() {
            k += 1;
        }
        let start = k;
        while k < q.len() && q.as_bytes()[k].is_ascii_digit() {
            k += 1;
        }
        if k > start {
            if let Ok(v) = q[start..k].parse::<usize>() {
                return Some(v);
            }
        }
    }

    // "top NN"
    if let Some(pos) = q.find("top ") {
        let mut k = pos + "top ".len();
        let start = k;
        while k < q.len() && q.as_bytes()[k].is_ascii_digit() {
            k += 1;
        }
        if k > start {
            if let Ok(v) = q[start..k].parse::<usize>() {
                return Some(v);
            }
        }
    }

    None
}

/// Deterministic trace compiler: natural-language-ish query -> semtrace JSON.
/// This intentionally supports a *small*, auditable subset.
/// Everything else must be provided as explicit JSON ops.
pub fn compile_query_to_trace(query: &str) -> Result<Trace> {
    let q = query.trim();
    if q.is_empty() {
        return Err(anyhow!("empty query"));
    }

    // QE (fractions) path: requires a fraction seed.
    if let Some(fr) = first_frac(q) {
        let mut ops: Vec<Op> = Vec::new();

        // Start at elem
        ops.push(Op::StartElem { elem: fr.clone() });

        // Optional: den<=6 constraint => bit 2 = 1 (per semtrace::bit_legend())
        if contains_den_le_6(q) {
            ops.push(Op::SetBit { i: 2, b: 1 });
        }

        // Always: nearest witness to the seed fraction
        ops.push(Op::WitnessNearest {
            target_elem: fr,
            metric: "ABS_DIFF".to_string(),
        });

        // Return set (sample)
        let max_items = parse_max_items(q).unwrap_or(20);
        let include_witness = contains_include_witness(q);
        ops.push(Op::ReturnSet {
            max_items,
            include_witness,
        });

        return Ok(Trace {
            semtrace_version: "0.0.1".to_string(),
            universe: "QE".to_string(),
            bits: 7,
            ops,
        });
    }

    // BOOLFUN path (very small subset): requires explicit universe + n + target.
    // Example supported forms:
    //   "BOOLFUN n=4 target=0xBEEF top 5"
    //   "boolfun n=4 target=bin:0101... k=10"
    //
    // Note: we do NOT attempt to infer n from arbitrary prose.
    let ql = q.to_ascii_lowercase();
    if ql.contains("boolfun") {
        let mut n: Option<u8> = None;
        let mut target: Option<String> = None;
        let mut k: Option<usize> = None;

        // n=NN
        if let Some(pos) = ql.find("n=") {
            let start = pos + 2;
            let mut end = start;
            let b = ql.as_bytes();
            while end < b.len() && b[end].is_ascii_digit() {
                end += 1;
            }
            if end > start {
                if let Ok(v) = ql[start..end].parse::<u8>() {
                    n = Some(v);
                }
            }
        }

        // target=...
        // capture until whitespace
        if let Some(pos) = ql.find("target=") {
            let start = pos + "target=".len();
            let mut end = start;
            let b = q.as_bytes();
            while end < b.len() && !b[end].is_ascii_whitespace() {
                end += 1;
            }
            if end > start {
                target = Some(q[start..end].to_string());
            }
        }

        // k=NN
        if let Some(pos) = ql.find("k=") {
            let start = pos + 2;
            let mut end = start;
            let b = ql.as_bytes();
            while end < b.len() && b[end].is_ascii_digit() {
                end += 1;
            }
            if end > start {
                if let Ok(v) = ql[start..end].parse::<usize>() {
                    k = Some(v);
                }
            }
        }

        // also accept "top NN"
        if k.is_none() {
            k = parse_max_items(q);
        }

        let n = n.ok_or_else(|| anyhow!("BOOLFUN compile requires n=..."))?;
        let target = target.ok_or_else(|| anyhow!("BOOLFUN compile requires target=..."))?;
        let k = k.unwrap_or(10);

        let ops: Vec<Op> = vec![
            Op::SelectUniverse {
                universe: "BOOLFUN".to_string(),
                n,
            },
            Op::TopK { target_elem: target, k },
            Op::ReturnSet {
                max_items: k,
                include_witness: true,
            },
        ];

        return Ok(Trace {
            semtrace_version: "0.0.1".to_string(),
            universe: "BOOLFUN".to_string(),
            bits: 7,
            ops,
        });
    }

    Err(anyhow!(
        "unable to deterministically compile query; provide explicit JSON ops"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compile_qe_den_le_6() {
        let t = compile_query_to_trace(
            "Find fractions similar to 7/200 but with denominator <= 6",
        )
        .unwrap();
        assert_eq!(t.universe, "QE");
        assert_eq!(t.bits, 7);
        assert!(matches!(t.ops[0], Op::StartElem { .. }));
        // must include den<=6 => SetBit i=2 b=1
        assert!(t
            .ops
            .iter()
            .any(|op| matches!(op, Op::SetBit { i: 2, b: 1 })));
        // must include WitnessNearest
        assert!(t.ops.iter().any(|op| matches!(op, Op::WitnessNearest { .. })));
        // must end with ReturnSet
        assert!(matches!(t.ops.last().unwrap(), Op::ReturnSet { .. }));
    }

    #[test]
    fn compile_qe_default_max_items() {
        let t = compile_query_to_trace("closest to 13/37").unwrap();
        let rs = t
            .ops
            .iter()
            .find_map(|op| {
                if let Op::ReturnSet {
                    max_items,
                    include_witness,
                } = op
                {
                    Some((*max_items, *include_witness))
                } else {
                    None
                }
            })
            .unwrap();
        assert_eq!(rs.0, 20);
        assert_eq!(rs.1, false);
    }

    #[test]
    fn compile_boolfun_requires_n_and_target() {
        let e = compile_query_to_trace("boolfun target=0xBEEF").err().unwrap();
        let msg = format!("{e}");
        assert!(msg.contains("n="));
        let e = compile_query_to_trace("boolfun n=4").err().unwrap();
        let msg = format!("{e}");
        assert!(msg.contains("target="));
    }

    #[test]
    fn compile_boolfun_ok() {
        let t = compile_query_to_trace("BOOLFUN n=4 target=0xBEEF k=5").unwrap();
        assert_eq!(t.universe, "BOOLFUN");
        assert!(t.ops.iter().any(|op| matches!(
            op,
            Op::SelectUniverse { universe, n } if universe=="BOOLFUN" && *n==4
        )));
        assert!(t.ops.iter().any(|op| matches!(
            op,
            Op::TopK { target_elem, k } if target_elem=="0xBEEF" && *k==5
        )));
    }
}
