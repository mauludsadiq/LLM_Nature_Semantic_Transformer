use std::cmp::Ordering;

/// Triangle with canonical ordering a ≤ b ≤ c
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct Tri {
    pub a: i32,
    pub b: i32,
    pub c: i32,
}

impl Tri {
    pub fn new(mut a: i32, mut b: i32, mut c: i32) -> Option<Self> {
        if a <= 0 || b <= 0 || c <= 0 {
            return None;
        }

        // sort a ≤ b ≤ c
        if a > b { std::mem::swap(&mut a, &mut b); }
        if b > c { std::mem::swap(&mut b, &mut c); }
        if a > b { std::mem::swap(&mut a, &mut b); }

        // triangle inequality
        if a + b <= c {
            return None;
        }

        Some(Tri { a, b, c })
    }

    /// canonical bytes (12 bytes, big-endian i32)
    #[allow(dead_code)]
    pub fn to_bytes(&self) -> [u8; 12] {
        let mut out = [0u8; 12];
        out[0..4].copy_from_slice(&self.a.to_be_bytes());
        out[4..8].copy_from_slice(&self.b.to_be_bytes());
        out[8..12].copy_from_slice(&self.c.to_be_bytes());
        out
    }

    pub fn perimeter(&self) -> i32 {
        self.a + self.b + self.c
    }

    pub fn is_isosceles(&self) -> bool {
        self.a == self.b || self.b == self.c
    }

    pub fn is_equilateral(&self) -> bool {
        self.a == self.b && self.b == self.c
    }

    pub fn is_primitive(&self) -> bool {
        gcd3(self.a, self.b, self.c) == 1
    }

    pub fn angle_type(&self) -> Ordering {
        // compare a^2 + b^2 vs c^2
        let lhs = (self.a * self.a) + (self.b * self.b);
        let rhs = self.c * self.c;
        lhs.cmp(&rhs)
    }
}

/// canonical ordering (like QE canonical_cmp)
pub fn canonical_cmp(a: &Tri, b: &Tri) -> Ordering {
    let pa = a.perimeter();
    let pb = b.perimeter();
    if pa != pb {
        return pa.cmp(&pb);
    }

    if a.a != b.a { return a.a.cmp(&b.a); }
    if a.b != b.b { return a.b.cmp(&b.b); }
    a.c.cmp(&b.c)
}

/// Build G_E universe (bounded)
pub fn build_ge(max_side: i32) -> Vec<Tri> {
    let mut out = Vec::new();

    for a in 1..=max_side {
        for b in a..=max_side {
            for c in b..=max_side {
                if let Some(t) = Tri::new(a, b, c) {
                    out.push(t);
                }
            }
        }
    }

    out.sort_by(canonical_cmp);
    out
}

/// distance for witness (L1)
#[allow(dead_code)]
pub fn tri_distance(a: &Tri, b: &Tri) -> i64 {
    ((a.a - b.a).abs()
        + (a.b - b.b).abs()
        + (a.c - b.c).abs()) as i64
}

/// helpers
fn gcd(mut a: i32, mut b: i32) -> i32 {
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    if a == 0 { 1 } else { a.abs() }
}

fn gcd3(a: i32, b: i32, c: i32) -> i32 {
    gcd(gcd(a, b), c)
}
