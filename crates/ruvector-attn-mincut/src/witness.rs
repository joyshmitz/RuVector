use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// A single witness entry for determinism verification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WitnessEntry {
    pub q_hash: String,
    pub k_hash: String,
    pub keep_mask: Vec<bool>,
    pub cut_cost: f32,
    pub lambda: f32,
    pub tau: usize,
    pub eps: f32,
    pub timestamp: u64,
}

/// Serialize a witness entry to a single JSONL line.
pub fn witness_log(entry: &WitnessEntry) -> String {
    serde_json::to_string(entry).unwrap_or_else(|_| "{}".to_string())
}

/// Compute SHA-256 hash of a float tensor, returned as a hex string.
///
/// The tensor is hashed by converting each f32 to its little-endian byte
/// representation and feeding the bytes into SHA-256.
pub fn hash_tensor(data: &[f32]) -> String {
    let mut hasher = Sha256::new();
    for &val in data {
        hasher.update(val.to_le_bytes());
    }
    let result = hasher.finalize();
    hex_encode(&result)
}

/// Simple hex encoding without pulling in the `hex` crate.
fn hex_encode(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        s.push_str(&format!("{:02x}", b));
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_tensor_deterministic() {
        let data = vec![1.0f32, 2.0, 3.0];
        let h1 = hash_tensor(&data);
        let h2 = hash_tensor(&data);
        assert_eq!(h1, h2);
        assert_eq!(h1.len(), 64); // SHA-256 = 32 bytes = 64 hex chars
    }

    #[test]
    fn test_hash_tensor_different_data() {
        let h1 = hash_tensor(&[1.0, 2.0]);
        let h2 = hash_tensor(&[1.0, 3.0]);
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_witness_log_roundtrip() {
        let entry = WitnessEntry {
            q_hash: "abc123".to_string(),
            k_hash: "def456".to_string(),
            keep_mask: vec![true, false, true],
            cut_cost: 1.5,
            lambda: 0.5,
            tau: 2,
            eps: 0.01,
            timestamp: 1000,
        };
        let json = witness_log(&entry);
        let restored: WitnessEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.q_hash, "abc123");
        assert_eq!(restored.keep_mask, vec![true, false, true]);
        assert!((restored.cut_cost - 1.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_hash_empty_tensor() {
        let h = hash_tensor(&[]);
        // SHA-256 of empty input is the well-known constant
        assert_eq!(h.len(), 64);
    }
}
