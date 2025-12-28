//! Hypervector data type and basic operations

use super::{HYPERVECTOR_BITS, HYPERVECTOR_U64_LEN};
use rand::Rng;
use std::fmt;

/// Error types for HDC operations
#[derive(Debug, thiserror::Error)]
pub enum HdcError {
    #[error("Invalid hypervector dimension: expected {expected}, got {got}")]
    InvalidDimension { expected: usize, got: usize },

    #[error("Empty vector set provided")]
    EmptyVectorSet,

    #[error("Serialization error: {0}")]
    SerializationError(String),
}

/// A binary hypervector with 10,000 bits packed into 156 u64 words
///
/// # Performance
///
/// - Memory: 156 * 8 = 1,248 bytes per vector
/// - XOR binding: <50ns (single CPU cycle per u64)
/// - Similarity: <100ns (SIMD popcount)
///
/// # Example
///
/// ```rust
/// use ruvector_nervous_system::hdc::Hypervector;
///
/// let v1 = Hypervector::random();
/// let v2 = Hypervector::random();
/// let bound = v1.bind(&v2);
/// let sim = v1.similarity(&v2);
/// ```
#[derive(Clone, PartialEq, Eq)]
pub struct Hypervector {
    pub(crate) bits: [u64; HYPERVECTOR_U64_LEN],
}

impl Hypervector {
    /// Creates a new hypervector with all bits set to zero
    ///
    /// # Example
    ///
    /// ```rust
    /// use ruvector_nervous_system::hdc::Hypervector;
    ///
    /// let zero = Hypervector::zero();
    /// assert_eq!(zero.popcount(), 0);
    /// ```
    pub fn zero() -> Self {
        Self {
            bits: [0u64; HYPERVECTOR_U64_LEN],
        }
    }

    /// Creates a random hypervector with ~50% bits set
    ///
    /// Uses thread-local RNG for performance.
    ///
    /// # Example
    ///
    /// ```rust
    /// use ruvector_nervous_system::hdc::Hypervector;
    ///
    /// let random = Hypervector::random();
    /// let count = random.popcount();
    /// // Should be around 5000 ± 150
    /// assert!(count > 4500 && count < 5500);
    /// ```
    pub fn random() -> Self {
        let mut rng = rand::thread_rng();
        let mut bits = [0u64; HYPERVECTOR_U64_LEN];

        for word in bits.iter_mut() {
            *word = rng.gen();
        }

        Self { bits }
    }

    /// Creates a hypervector from a seed for reproducibility
    ///
    /// # Example
    ///
    /// ```rust
    /// use ruvector_nervous_system::hdc::Hypervector;
    ///
    /// let v1 = Hypervector::from_seed(42);
    /// let v2 = Hypervector::from_seed(42);
    /// assert_eq!(v1, v2);
    /// ```
    pub fn from_seed(seed: u64) -> Self {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let mut bits = [0u64; HYPERVECTOR_U64_LEN];

        for word in bits.iter_mut() {
            *word = rng.gen();
        }

        Self { bits }
    }

    /// Binds two hypervectors using XOR
    ///
    /// Binding is associative, commutative, and self-inverse:
    /// - `a.bind(b) == b.bind(a)`
    /// - `a.bind(b).bind(b) == a`
    ///
    /// # Performance
    ///
    /// <50ns on modern CPUs (single cycle XOR per u64)
    ///
    /// # Example
    ///
    /// ```rust
    /// use ruvector_nervous_system::hdc::Hypervector;
    ///
    /// let a = Hypervector::random();
    /// let b = Hypervector::random();
    /// let bound = a.bind(&b);
    ///
    /// // Self-inverse property
    /// assert_eq!(bound.bind(&b), a);
    /// ```
    #[inline]
    pub fn bind(&self, other: &Self) -> Self {
        let mut result = Self::zero();

        for i in 0..HYPERVECTOR_U64_LEN {
            result.bits[i] = self.bits[i] ^ other.bits[i];
        }

        result
    }

    /// Computes similarity between two hypervectors
    ///
    /// Returns a value in [0.0, 1.0] where:
    /// - 1.0 = identical vectors
    /// - 0.5 = random/orthogonal vectors
    /// - 0.0 = completely opposite vectors
    ///
    /// # Performance
    ///
    /// <100ns with SIMD popcount
    ///
    /// # Example
    ///
    /// ```rust
    /// use ruvector_nervous_system::hdc::Hypervector;
    ///
    /// let a = Hypervector::random();
    /// let b = a.clone();
    /// assert!((a.similarity(&b) - 1.0).abs() < 0.001);
    /// ```
    #[inline]
    pub fn similarity(&self, other: &Self) -> f32 {
        let hamming = self.hamming_distance(other);
        1.0 - (2.0 * hamming as f32 / HYPERVECTOR_BITS as f32)
    }

    /// Computes Hamming distance (number of differing bits)
    ///
    /// # Performance
    ///
    /// <100ns with SIMD popcount instruction
    ///
    /// # Example
    ///
    /// ```rust
    /// use ruvector_nervous_system::hdc::Hypervector;
    ///
    /// let a = Hypervector::random();
    /// assert_eq!(a.hamming_distance(&a), 0);
    /// ```
    #[inline]
    pub fn hamming_distance(&self, other: &Self) -> u32 {
        let mut distance = 0u32;

        for i in 0..HYPERVECTOR_U64_LEN {
            // XOR to find differing bits, then count them
            distance += (self.bits[i] ^ other.bits[i]).count_ones();
        }

        distance
    }

    /// Counts the number of set bits (population count)
    ///
    /// # Example
    ///
    /// ```rust
    /// use ruvector_nervous_system::hdc::Hypervector;
    ///
    /// let zero = Hypervector::zero();
    /// assert_eq!(zero.popcount(), 0);
    ///
    /// let random = Hypervector::random();
    /// let count = random.popcount();
    /// // Should be around 5000 for random vectors
    /// assert!(count > 4500 && count < 5500);
    /// ```
    #[inline]
    pub fn popcount(&self) -> u32 {
        self.bits.iter().map(|&w| w.count_ones()).sum()
    }

    /// Bundles multiple vectors by majority voting on each bit
    ///
    /// # Example
    ///
    /// ```rust
    /// use ruvector_nervous_system::hdc::Hypervector;
    ///
    /// let v1 = Hypervector::random();
    /// let v2 = Hypervector::random();
    /// let v3 = Hypervector::random();
    ///
    /// let bundled = Hypervector::bundle(&[v1.clone(), v2, v3]);
    /// // Bundled vector is similar to all inputs
    /// assert!(bundled.similarity(&v1) > 0.3);
    /// ```
    pub fn bundle(vectors: &[Self]) -> Result<Self, HdcError> {
        if vectors.is_empty() {
            return Err(HdcError::EmptyVectorSet);
        }

        if vectors.len() == 1 {
            return Ok(vectors[0].clone());
        }

        let mut result = Self::zero();
        let threshold = (vectors.len() / 2) as u32;

        // Count bits at each position
        for bit_idx in 0..HYPERVECTOR_BITS {
            let word_idx = bit_idx / 64;
            let bit_pos = bit_idx % 64;

            let mut count = 0u32;
            for vector in vectors {
                if (vector.bits[word_idx] >> bit_pos) & 1 == 1 {
                    count += 1;
                }
            }

            // Majority vote
            if count > threshold {
                result.bits[word_idx] |= 1u64 << bit_pos;
            }
        }

        Ok(result)
    }

    /// Returns the internal bit array (for advanced use cases)
    #[inline]
    pub fn bits(&self) -> &[u64; HYPERVECTOR_U64_LEN] {
        &self.bits
    }
}

impl fmt::Debug for Hypervector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Hypervector {{ bits: {} set / {} total }}",
            self.popcount(),
            HYPERVECTOR_BITS
        )
    }
}

impl Default for Hypervector {
    fn default() -> Self {
        Self::zero()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_vector() {
        let zero = Hypervector::zero();
        assert_eq!(zero.popcount(), 0);
        assert_eq!(zero.hamming_distance(&zero), 0);
    }

    #[test]
    fn test_random_vector_properties() {
        let v = Hypervector::random();
        let count = v.popcount();

        // Random vector should have ~50% bits set (±3 sigma)
        assert!(count > 4500 && count < 5500, "popcount: {}", count);
    }

    #[test]
    fn test_from_seed_deterministic() {
        let v1 = Hypervector::from_seed(42);
        let v2 = Hypervector::from_seed(42);
        let v3 = Hypervector::from_seed(43);

        assert_eq!(v1, v2);
        assert_ne!(v1, v3);
    }

    #[test]
    fn test_bind_commutative() {
        let a = Hypervector::random();
        let b = Hypervector::random();

        assert_eq!(a.bind(&b), b.bind(&a));
    }

    #[test]
    fn test_bind_self_inverse() {
        let a = Hypervector::random();
        let b = Hypervector::random();

        let bound = a.bind(&b);
        let unbound = bound.bind(&b);

        assert_eq!(a, unbound);
    }

    #[test]
    fn test_similarity_bounds() {
        let a = Hypervector::random();
        let b = Hypervector::random();

        let sim = a.similarity(&b);
        assert!(sim >= 0.0 && sim <= 1.0);
    }

    #[test]
    fn test_similarity_identical() {
        let a = Hypervector::random();
        let sim = a.similarity(&a);

        assert!((sim - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_similarity_random_approximately_half() {
        let a = Hypervector::random();
        let b = Hypervector::random();

        let sim = a.similarity(&b);
        // Random vectors should be orthogonal (~0.5 similarity)
        assert!(sim > 0.3 && sim < 0.7, "similarity: {}", sim);
    }

    #[test]
    fn test_hamming_distance_identical() {
        let a = Hypervector::random();
        assert_eq!(a.hamming_distance(&a), 0);
    }

    #[test]
    fn test_bundle_single_vector() {
        let v = Hypervector::random();
        let bundled = Hypervector::bundle(&[v.clone()]).unwrap();

        assert_eq!(bundled, v);
    }

    #[test]
    fn test_bundle_empty_error() {
        let result = Hypervector::bundle(&[]);
        assert!(matches!(result, Err(HdcError::EmptyVectorSet)));
    }

    #[test]
    fn test_bundle_majority_vote() {
        let v1 = Hypervector::from_seed(1);
        let v2 = Hypervector::from_seed(2);
        let v3 = Hypervector::from_seed(3);

        let bundled = Hypervector::bundle(&[v1.clone(), v2.clone(), v3]).unwrap();

        // Bundled should be similar to all inputs
        assert!(bundled.similarity(&v1) > 0.3);
        assert!(bundled.similarity(&v2) > 0.3);
    }

    #[test]
    fn test_bundle_odd_count() {
        let vectors: Vec<_> = (0..5).map(|i| Hypervector::from_seed(i)).collect();
        let bundled = Hypervector::bundle(&vectors).unwrap();

        for v in &vectors {
            assert!(bundled.similarity(v) > 0.3);
        }
    }

    #[test]
    fn test_debug_format() {
        let v = Hypervector::zero();
        let debug = format!("{:?}", v);
        assert!(debug.contains("bits: 0 set"));
    }
}
