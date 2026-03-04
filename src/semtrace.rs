use crate::qe::Frac;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Trace {
    pub semtrace_version: String,
    pub universe: String,
    pub bits: u8,
    pub ops: Vec<Op>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "op")]
pub enum Op {
    #[serde(rename = "START_ELEM")]
    StartElem { elem: String },
    #[serde(rename = "SET_BIT")]
    SetBit { i: u8, b: u8 },
    #[serde(rename = "SELECT_UNIVERSE")]
    SelectUniverse { universe: String, n: u8 },
    #[serde(rename = "FILTER_WEIGHT")]
    FilterWeight { min: u32, max: u32 },
    #[serde(rename = "TOPK")]
    TopK { target_elem: String, k: usize },
    #[serde(rename = "WITNESS_NEAREST")]
    WitnessNearest { target_elem: String, metric: String },
    #[serde(rename = "RETURN_SET")]
    ReturnSet {
        max_items: usize,
        include_witness: bool,
    },
    #[serde(rename = "JOIN_NEAREST")]
    JoinNearest {
        left_universe: String,
        right_universe: String,
        left_elem: String,
        right_elem: String,
        metric: String,
    },
}

#[allow(dead_code)]

pub fn read_trace_json(path: &Path) -> anyhow::Result<Trace> {
    let txt = fs::read_to_string(path)?;
    let t: Trace = serde_json::from_str(&txt)?;
    Ok(t)
}

/// Deterministic v0 demo trace: start at 7/200, force den<=6=1 (bit index 2),
/// pick nearest witness, and return a sample.
#[allow(dead_code)]
pub fn demo_trace() -> Trace {
    Trace {
        semtrace_version: "0.0.1".to_string(),
        universe: "QE".to_string(),
        bits: 7,
        ops: vec![
            Op::StartElem {
                elem: "7/200".to_string(),
            },
            Op::SetBit { i: 2, b: 1 },
            Op::WitnessNearest {
                target_elem: "7/200".to_string(),
                metric: "ABS_DIFF".to_string(),
            },
            Op::ReturnSet {
                max_items: 20,
                include_witness: true,
            },
        ],
    }
}

/// For v0: map bit index to predicate meaning (QE fixed).
pub fn bit_legend() -> [&'static str; 7] {
    [
        "positive",
        "rat_int",
        "den<=6",
        "num_even",
        "den_mod3",
        "proper",
        "num_abs<=5",
    ]
}

pub fn bit_legend_geom() -> [&'static str; 7] {
    [
        "perim<=20",
        "isosceles",
        "equilateral",
        "primitive",
        "right",
        "acute",
        "obtuse",
    ]
}

/// Compute signature bits for QE predicates.
pub fn sig7(f: &Frac) -> u8 {
    let mut bits: u8 = 0;
    let positive = f.num > 0;
    let integer = f.den == 1; // true iff the fraction is a whole number
    let den_le_6 = f.den <= 6;
    let num_even = f.num % 2 == 0;
    let den_mod3 = f.den % 3 == 0;
    let proper = f.num.abs() < f.den;
    let num_abs_le_5 = f.num.abs() <= 5;

    let preds = [
        positive,
        integer,
        den_le_6,
        num_even,
        den_mod3,
        proper,
        num_abs_le_5,
    ];
    for (i, p) in preds.iter().enumerate() {
        if *p {
            bits |= 1u8 << i;
        }
    }
    bits
}

// ---------------- GEOMETRY SIGNATURE ----------------
use crate::geom::Tri;

pub fn sig7_geom(t: &Tri) -> u8 {
    let mut bits: u8 = 0;

    let perimeter_le_20 = t.perimeter() <= 20; // bit 0
    let is_isosceles = t.is_isosceles(); // bit 1
    let is_equilateral = t.is_equilateral(); // bit 2
    let is_primitive = t.is_primitive(); // bit 3
    let is_right = t.angle_type() == std::cmp::Ordering::Equal; // bit 4
    let is_acute = t.angle_type() == std::cmp::Ordering::Greater; // bit 5
    let is_obtuse = t.angle_type() == std::cmp::Ordering::Less; // bit 6

    let preds = [
        perimeter_le_20,
        is_isosceles,
        is_equilateral,
        is_primitive,
        is_right,
        is_acute,
        is_obtuse,
    ];

    for (i, p) in preds.iter().enumerate() {
        if *p {
            bits |= 1u8 << i;
        }
    }

    bits
}

/// Constraint (mask,value) for partial signature filtering.
#[derive(Clone, Copy, Debug)]
pub struct Constraint {
    pub mask: u8,
    pub value: u8,
}

impl Constraint {
    pub fn empty() -> Self {
        Constraint { mask: 0, value: 0 }
    }
    pub fn set_bit(mut self, i: u8, b: u8) -> Self {
        let bit = 1u8 << i;
        self.mask |= bit;
        if b == 1 {
            self.value |= bit;
        } else {
            self.value &= !bit;
        }
        self
    }
    pub fn matches(&self, sig: u8) -> bool {
        (sig & self.mask) == (self.value & self.mask)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qe::Frac;

    #[test]
    fn sig7_integer_fraction() {
        // Frac { num: 3, den: 1 } is a positive integer
        let f = Frac { num: 3, den: 1 };
        let s = sig7(&f);
        // bit 0: positive (3 > 0) = 1
        assert_eq!((s >> 0) & 1, 1, "positive");
        // bit 1: rat_int (den == 1) = 1
        assert_eq!((s >> 1) & 1, 1, "rat_int");
        // bit 2: den<=6 (1 <= 6) = 1
        assert_eq!((s >> 2) & 1, 1, "den<=6");
        // bit 3: num_even (3 % 2 == 0) = 0
        assert_eq!((s >> 3) & 1, 0, "num_even");
        // bit 4: den_mod3 (1 % 3 == 0) = 0
        assert_eq!((s >> 4) & 1, 0, "den_mod3");
        // bit 5: proper (3.abs() < 1) = 0
        assert_eq!((s >> 5) & 1, 0, "proper");
        // bit 6: num_abs<=5 (3 <= 5) = 1
        assert_eq!((s >> 6) & 1, 1, "num_abs<=5");
        // stable literal: 0b1000111 = 71
        assert_eq!(s, 0b1000111, "stable signature for Frac {{ num:3, den:1 }}");
    }

    #[test]
    fn sig7_negative_integer() {
        let f = Frac { num: -2, den: 1 };
        let s = sig7(&f);
        // bit 0: positive (-2 > 0) = 0
        assert_eq!((s >> 0) & 1, 0, "positive");
        // bit 1: rat_int (den == 1) = 1
        assert_eq!((s >> 1) & 1, 1, "rat_int");
        // bit 2: den<=6 = 1
        assert_eq!((s >> 2) & 1, 1, "den<=6");
        // bit 3: num_even (-2 % 2 == 0) = 1
        assert_eq!((s >> 3) & 1, 1, "num_even");
        // bit 4: den_mod3 (1 % 3 == 0) = 0
        assert_eq!((s >> 4) & 1, 0, "den_mod3");
        // bit 5: proper (2 < 1) = 0
        assert_eq!((s >> 5) & 1, 0, "proper");
        // bit 6: num_abs<=5 (2 <= 5) = 1
        assert_eq!((s >> 6) & 1, 1, "num_abs<=5");
        assert_eq!(s, 0b1001110, "stable signature for Frac {{ num:-2, den:1 }}");
    }
}
