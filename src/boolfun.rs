use serde::{Deserialize, Serialize};
use std::cmp::Ordering;

pub fn is_boolfun_universe(u_norm: &str) -> bool {
    matches!(
        u_norm,
        "BOOLFUN"
            | "BOOLFUN<N>"
            | "BOOLFUN4"
            | "BOOLFUN_4"
            | "BOOLFUNV0"
            | "BOOLFUNV1"
            | "BOOLFUNS"
            | "BOOLFUNS<N>"
            | "BOOLFUNS4"
            | "BOOLFUNS_4"
            | "BOOLFUNS_V0"
            | "BOOLFUNS_V1"
            | "BOOLFUN7"
            | "BOOLFUN_7"
            | "BOOLFUN<N=7>"
            | "BOOLFUNSIG7"
            | "BOOLFUNSIG_7"
    )
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, Eq, PartialEq, Hash)]
pub struct BoolFun {
    pub n: u8,     // number of input vars
    pub bits: u64, // packed truth-table outputs, LSB = input 0..0
}

impl BoolFun {
    /// Number of rows in truth table = 2^n.
    pub fn rows(&self) -> u32 {
        // Special universe: n=7 represents a 7-bit signature element (128 elems), not a 2^n truth-table.
        if self.n == 7 {
            return 7;
        }
        1u32 << (self.n as u32)
    }

    /// Mask of valid output bits (low 2^n bits).
    pub fn mask(&self) -> u64 {
        // n=7 signature universe: only low 7 bits are meaningful.
        if self.n == 7 {
            return 0x7f;
        }
        let r = self.rows();
        if r == 64 {
            u64::MAX
        } else {
            (1u64 << r) - 1
        }
    }

    /// Canonical bytes for hashing/merkle: fixed 9 bytes = [n:u8] + [bits:u64 BE].
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(1 + 8);
        out.push(self.n);
        out.extend_from_slice(&self.bits.to_be_bytes());
        out
    }

    /// Hamming weight (number of 1 outputs) in the truth table.
    pub fn weight(&self) -> u32 {
        (self.bits & self.mask()).count_ones()
    }

    /// Hamming distance between two functions (requires same n).
    pub fn hamming(&self, other: &Self) -> u32 {
        if self.n != other.n {
            // Treat as incomparable; caller should enforce same universe.
            return u32::MAX;
        }
        ((self.bits ^ other.bits) & self.mask()).count_ones()
    }
}

/// Canonical total order: (n ascending, bits ascending).
pub fn canonical_cmp(a: &BoolFun, b: &BoolFun) -> Ordering {
    let o1 = a.n.cmp(&b.n);
    if o1 != Ordering::Equal {
        return o1;
    }
    a.bits.cmp(&b.bits)
}

/// Build BoolFun universe:
/// - For n=4: 2^(2^4)=65536 functions => packed bits are 16-bit columns.
/// - For smaller n: generate full space 0..2^(2^n)-1.
/// Note: capped at n<=6 (64 rows) to keep bits in u64.
pub fn build_boolfun(n: u8) -> Vec<BoolFun> {
    // Special universe: n=7 is a 7-bit signature space (0..127), not the full 2^(2^n) truth-table space.
    if n == 7 {
        let mut v: Vec<BoolFun> = Vec::with_capacity(128);
        for bits in 0u64..128u64 {
            v.push(BoolFun { n, bits });
        }
        v.sort_by(canonical_cmp);
        return v;
    }

    let rows = 1u32 << (n as u32);
    assert!(rows <= 64, "BoolFun n too large for u64 packing");
    let total: u64 = if rows == 64 { u64::MAX } else { 1u64 << rows };

    let mut v: Vec<BoolFun> = Vec::with_capacity(total as usize);
    for bits in 0..total {
        v.push(BoolFun { n, bits });
    }
    v.sort_by(canonical_cmp);
    v
}

/// Parse element encodings:
/// - "0xBEEF" (hex, implies n=4)
/// - "u16:48879" (decimal, implies n=4)
/// - "bin:0101..." (length must be 2^n; n inferred from length)
///
/// Returns BoolFun with (n,bits) normalized to packed LSB-first ordering:
/// For bin: string is interpreted left-to-right as MSB..LSB of the packed value,
/// so "bin:0001" => bits=1 (LSB=1).
pub fn parse_elem(s: &str) -> Option<BoolFun> {
    let t = s.trim();

    if let Some(hexs) = t.strip_prefix("0x").or_else(|| t.strip_prefix("0X")) {
        let bits = u64::from_str_radix(hexs.trim(), 16).ok()?;
        return Some(BoolFun {
            n: 4,
            bits: bits & 0xFFFF,
        });
    }

    if let Some(ds) = t.strip_prefix("u16:") {
        let bits: u64 = ds.trim().parse::<u64>().ok()?;
        return Some(BoolFun {
            n: 4,
            bits: bits & 0xFFFF,
        });
    }

    if let Some(ds) = t.strip_prefix("u64:").or_else(|| t.strip_prefix("u8:")) {
        let bits: u64 = ds.trim().parse::<u64>().ok()?;
        if bits > 0x7f {
            return None;
        }
        return Some(BoolFun { n: 7, bits: bits & 0x7f });
    }

    if let Some(b0) = t.strip_prefix("0b").or_else(|| t.strip_prefix("0B")) {
        let b = b0.trim();
        if b.is_empty() {
            return None;
        }
        if !b.chars().all(|c| c == '0' || c == '1') {
            return None;
        }
        // 0b... parsing:
        // - if <=7 bits: signature element (n=7, bits in 0..127)
        // - else (<=16 bits): packed u16 truth-table (n=4), MSB..LSB
        if b.len() > 16 {
            return None;
        }
        let mut bits: u64 = 0;
        for ch in b.chars() {
            bits <<= 1;
            if ch == '1' {
                bits |= 1;
            }
        }
        if b.len() <= 7 {
            return Some(BoolFun {
                n: 7,
                bits: bits & 0x7f,
            });
        }
        return Some(BoolFun {
            n: 4,
            bits: bits & 0xFFFF,
        });
    }
    if let Some(bs) = t.strip_prefix("bin:") {
        let b = bs.trim();
        if b.is_empty() {
            return None;
        }
        if !b.chars().all(|c| c == '0' || c == '1') {
            return None;
        }

        let len = b.len() as u32;
        // len must be a power of two: len = 2^n
        if len == 0 || (len & (len - 1)) != 0 {
            return None;
        }

        let n = (len as f64).log2() as u8;
        if (1u32 << (n as u32)) != len {
            return None;
        }
        if len > 64 {
            return None;
        }

        // Interpret string as MSB..LSB of packed value
        let mut bits: u64 = 0;
        for ch in b.chars() {
            bits <<= 1;
            if ch == '1' {
                bits |= 1;
            }
        }
        return Some(BoolFun { n, bits });
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn boolfun_n4_size() {
        let v = build_boolfun(4);
        assert_eq!(v.len(), 65536);
        assert_eq!(v[0], BoolFun { n: 4, bits: 0 });
        assert_eq!(v.last().unwrap(), &BoolFun { n: 4, bits: 65535 });
    }

    #[test]
    fn parse_hex_u16() {
        let a = parse_elem("0xBEEF").unwrap();
        let b = parse_elem("u16:48879").unwrap();
        assert_eq!(a.n, 4);
        assert_eq!(a, b);
        assert_eq!(a.bits, 0xBEEF);
    }

    #[test]
    fn parse_bin_infers_n() {
        let f = parse_elem("bin:0001").unwrap();
        assert_eq!(f.n, 2);
        assert_eq!(f.bits, 1);
        assert_eq!(f.weight(), 1);
    }

    #[test]
    fn parse_0b_signature_n7() {
        let f = parse_elem("0b1010101").unwrap();
        assert_eq!(f.n, 7);
        assert_eq!(f.bits, 0b1010101);
        assert_eq!(f.mask(), 0x7f);
    }
    #[test]
    fn parse_u64_prefix() {
        // u64: is a 7-bit QE signature format (n=7, bits 0..=127)
        let f = parse_elem("u64:35").unwrap();
        assert_eq!(f.bits, 35);
        assert_eq!(f.n, 7);
        // u64:0 edge case
        let z = parse_elem("u64:0").unwrap();
        assert_eq!(z.bits, 0);
        assert_eq!(z.n, 7);
        // u64:127 is max valid value
        let max = parse_elem("u64:127").unwrap();
        assert_eq!(max.bits, 127);
        assert_eq!(max.n, 7);
        // u64:128 exceeds 7 bits, rejected
        assert!(parse_elem("u64:128").is_none());
    }

}
