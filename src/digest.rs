use sha2::{Digest, Sha256};

pub fn sha256_bytes(data: &[u8]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(data);
    let out = h.finalize();
    let mut b = [0u8; 32];
    b.copy_from_slice(&out);
    b
}

pub fn sha256_hex(data: &[u8]) -> String {
    hex::encode(sha256_bytes(data))
}

/// Merkle root over leaves that are already 32-byte hashes.
/// If leaves are empty, returns sha256("") (defined root).
pub fn merkle_root(leaves: &[[u8; 32]]) -> [u8; 32] {
    if leaves.is_empty() {
        return sha256_bytes(b"");
    }
    let mut level: Vec<[u8; 32]> = leaves.to_vec();
    while level.len() > 1 {
        let mut next = Vec::with_capacity((level.len() + 1) / 2);
        let mut i = 0;
        while i < level.len() {
            let left = level[i];
            let right = if i + 1 < level.len() { level[i + 1] } else { level[i] };
            let mut buf = [0u8; 64];
            buf[0..32].copy_from_slice(&left);
            buf[32..64].copy_from_slice(&right);
            next.push(sha256_bytes(&buf));
            i += 2;
        }
        level = next;
    }
    level[0]
}
