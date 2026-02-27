use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::BTreeSet;

#[derive(Clone, Copy, Debug, Serialize, Deserialize, Eq, PartialEq, Hash)]
pub struct Frac {
    pub num: i32,
    pub den: i32, // always >0
}

fn gcd(mut a: i32, mut b: i32) -> i32 {
    a = a.abs();
    b = b.abs();
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    if a == 0 { 1 } else { a }
}

impl Frac {
    pub fn new_reduced(num: i32, den: i32) -> Self {
        assert!(den != 0);
        let mut n = num;
        let mut d = den;
        if d < 0 { n = -n; d = -d; }
        let g = gcd(n, d);
        Frac { num: n / g, den: d / g }
    }

    /// Canonical bytes for hashing/merkle: fixed 8 bytes (i32 num, i32 den) big-endian.
    pub fn canonical_bytes(&self) -> [u8; 8] {
        let mut out = [0u8; 8];
        out[0..4].copy_from_slice(&self.num.to_be_bytes());
        out[4..8].copy_from_slice(&self.den.to_be_bytes());
        out
    }

    /// Compare by numeric value exactly via cross-multiply.
    pub fn cmp_value(&self, other: &Frac) -> Ordering {
        // compare self.num/self.den ? other.num/other.den using i64 to avoid overflow
        let a = self.num as i64;
        let b = self.den as i64;
        let c = other.num as i64;
        let d = other.den as i64;
        (a * d).cmp(&(c * b))
    }

    pub fn abs_num(&self) -> i32 { self.num.abs() }
}

/// Canonical total order used everywhere (sets, merkle leaves, witness tie-breaks):
/// 1) numeric value (exact) via cross-multiply
/// 2) |numerator| ascending
/// 3) denominator ascending
/// 4) sign (negative < positive)
pub fn canonical_cmp(a: &Frac, b: &Frac) -> Ordering {
    let o1 = a.cmp_value(b);
    if o1 != Ordering::Equal { return o1; }
    let o2 = a.abs_num().cmp(&b.abs_num());
    if o2 != Ordering::Equal { return o2; }
    let o3 = a.den.cmp(&b.den);
    if o3 != Ordering::Equal { return o3; }
    a.num.cmp(&b.num) // negative < positive if everything else equal
}

/// Build QE exactly:
/// denominators 1..=200, numerators -200..=200, reduced to unique fractions.
pub fn build_qe() -> Vec<Frac> {
    let mut set: BTreeSet<(i32, i32)> = BTreeSet::new();
    for den in 1..=200 {
        for num in -200..=200 {
            let f = Frac::new_reduced(num, den);
            set.insert((f.num, f.den));
        }
    }
    let mut v: Vec<Frac> = set.into_iter().map(|(n,d)| Frac{num:n,den:d}).collect();
    v.sort_by(canonical_cmp);
    v
}

/// Parse "a/b" into reduced Frac.
pub fn parse_frac(s: &str) -> Option<Frac> {
    let parts: Vec<&str> = s.trim().split('/').collect();
    if parts.len() != 2 { return None; }
    let num: i32 = parts[0].parse().ok()?;
    let den: i32 = parts[1].parse().ok()?;
    if den == 0 { return None; }
    Some(Frac::new_reduced(num, den))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn qe_size_matches_certified() {
        let qe = build_qe();
        assert_eq!(qe.len(), 48927);
        assert_eq!(qe.last().unwrap(), &Frac { num: 200, den: 1 });
    }
}
