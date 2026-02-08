//! Block-based storage engine for temporal tensor compression (ADR-018).
//!
//! Provides tiered quantized storage with CRC32 integrity checking,
//! access-pattern tracking, and eviction support. Each block of tensor
//! data is quantized at the bit width appropriate for its storage tier
//! and tracked with rich metadata for tier-promotion/demotion decisions.
//!
//! # Storage Tiers
//!
//! | Tier  | Bits | Description                         |
//! |-------|------|-------------------------------------|
//! | Tier0 | 0    | Evicted: metadata only, no payload  |
//! | Tier1 | 8    | Hot: full fidelity quantization      |
//! | Tier2 | 7    | Warm: moderate compression           |
//! | Tier3 | 3    | Cold: aggressive compression         |
//!
//! # Example
//!
//! ```rust
//! use ruvector_temporal_tensor::store::{BlockKey, Tier, TieredStore, ReconstructPolicy};
//!
//! let mut store = TieredStore::new(4096);
//! let key = BlockKey { tensor_id: 1, block_index: 0 };
//! let data = vec![1.0f32; 64];
//!
//! store.put(key, &data, Tier::Tier1, 0).unwrap();
//! assert_eq!(store.block_count(), 1);
//!
//! let mut out = vec![0.0f32; 64];
//! let n = store.get(key, &mut out).unwrap();
//! assert_eq!(n, 64);
//! ```

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// Unique identifier for a tensor block.
///
/// Composed of the owning tensor's 128-bit ID and a block index within
/// that tensor, allowing fine-grained block-level storage and retrieval.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct BlockKey {
    pub tensor_id: u128,
    pub block_index: u32,
}

/// Storage tier for a block.
///
/// Tiers form a hierarchy from hot (high fidelity, fast access) to evicted
/// (metadata-only, zero payload bytes).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
#[repr(u8)]
pub enum Tier {
    /// Evicted: compressed to zero bits, only metadata remains.
    Tier0 = 0,
    /// Hot: 8-bit quantization, full fidelity.
    Tier1 = 1,
    /// Warm: 7-bit quantization.
    Tier2 = 2,
    /// Cold: 3-bit quantization.
    Tier3 = 3,
}

/// Data type of the original tensor.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum DType {
    F32 = 0,
    F16 = 1,
    BF16 = 2,
}

/// Reconstruction policy for evicted (Tier0) blocks.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum ReconstructPolicy {
    /// No reconstruction possible. Reads fail or return zeros.
    None = 0,
    /// Reconstruct from base + delta chain.
    Delta = 1,
    /// Reconstruct from stored low-rank factors.
    Factor = 2,
}

/// Complete metadata for a single block.
#[derive(Clone, Debug)]
pub struct BlockMeta {
    pub key: BlockKey,
    pub dtype: DType,
    pub tier: Tier,
    /// Quantization bit width (8, 7, 5, or 3).
    pub bits: u8,
    /// Quantization scale: `max(|v|) / qmax`.
    pub scale: f32,
    /// Quantization zero point (0 for symmetric).
    pub zero_point: i16,
    /// Tick at which this block was created.
    pub created_at: u64,
    /// Tick of the most recent access.
    pub last_access_at: u64,
    /// Cumulative access count.
    pub access_count: u32,
    /// Exponential moving average of access rate.
    pub ema_rate: f32,
    /// Sliding-window bitset for the last 64 ticks.
    pub window: u64,
    /// CRC32 checksum of quantized payload concatenated with scale bytes.
    pub checksum: u32,
    /// How to reconstruct if evicted.
    pub reconstruct: ReconstructPolicy,
    /// Number of ticks spent in the current tier.
    pub tier_age: u32,
    /// Optional parent tensor ID for delta-chain lineage.
    pub lineage_parent: Option<u128>,
    /// Size of this block's quantized payload in bytes.
    pub block_bytes: u32,
}

/// Errors produced by the storage engine.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StoreError {
    /// The block has been evicted to Tier0 and cannot be read directly.
    TensorEvicted,
    /// No block exists for the given key.
    BlockNotFound,
    /// CRC32 verification failed after read.
    ChecksumMismatch,
    /// An underlying I/O operation failed.
    IOError,
    /// The memory budget has been exhausted.
    BudgetExhausted,
    /// The block data is malformed or invalid.
    InvalidBlock,
    /// A delta reconstruction chain exceeded the maximum depth.
    DeltaChainTooLong,
    /// Reconstruction of an evicted block failed.
    ReconstructionFailed,
    /// The provided data is malformed or could not be parsed.
    InvalidData,
    /// The delta chain is at maximum length and cannot accept more deltas.
    ChainFull,
}

// ---------------------------------------------------------------------------
// Traits
// ---------------------------------------------------------------------------

/// Clock abstraction for deterministic time in tests and production.
pub trait Clock {
    /// Returns the current tick count.
    fn now_ticks(&self) -> u64;
}

/// Block I/O for reading and writing raw quantized data.
pub trait BlockIO {
    /// Read quantized bytes for `key` from the given `tier` into `dst`.
    /// Returns the number of bytes written to `dst`.
    fn read_block(&self, tier: Tier, key: BlockKey, dst: &mut [u8]) -> Result<usize, StoreError>;

    /// Write raw quantized bytes `src` for `key` into the given `tier`.
    fn write_block(&mut self, tier: Tier, key: BlockKey, src: &[u8]) -> Result<(), StoreError>;

    /// Delete the raw data for `key` from the given `tier`.
    fn delete_block(&mut self, tier: Tier, key: BlockKey) -> Result<(), StoreError>;
}

/// Metadata log for append-only persistence of block metadata.
pub trait MetaLog {
    /// Append (or upsert) a metadata record.
    fn append(&mut self, rec: &BlockMeta) -> Result<(), StoreError>;

    /// Look up metadata by key.
    fn get(&self, key: BlockKey) -> Option<&BlockMeta>;

    /// Iterate over all metadata records.
    fn iter(&self) -> Box<dyn Iterator<Item = &BlockMeta> + '_>;
}

// ---------------------------------------------------------------------------
// CRC32
// ---------------------------------------------------------------------------

/// Compute CRC32 using the standard reflected polynomial (0xEDB88320).
///
/// This is the same algorithm used by zlib/gzip/PNG. No lookup table is
/// used to keep the binary small; the byte-at-a-time loop is sufficient
/// for the block sizes involved.
pub fn crc32(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFF_FFFF;
    for &byte in data {
        crc ^= byte as u32;
        for _ in 0..8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ 0xEDB8_8320;
            } else {
                crc >>= 1;
            }
        }
    }
    !crc
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Return the default bit width for a storage tier.
fn bits_for_tier(tier: Tier) -> u8 {
    match tier {
        Tier::Tier0 => 0,
        Tier::Tier1 => 8,
        Tier::Tier2 => 7,
        Tier::Tier3 => 3,
    }
}

/// Compute the maximum representable signed magnitude for a given bit width.
///
/// `qmax = 2^(bits-1) - 1`. Returns 0 for invalid widths.
#[inline]
fn qmax(bits: u8) -> i32 {
    if bits == 0 || bits > 8 {
        return 0;
    }
    (1i32 << (bits - 1)) - 1
}

/// Internal representation of a stored quantized block.
struct BlockData {
    /// Number of original f32 elements (needed for exact dequantization).
    element_count: u32,
    /// Packed quantized bytes.
    packed: Vec<u8>,
}

/// Quantize an f32 slice using symmetric quantization at the given bit width.
///
/// Returns the packed byte vector and the computed scale factor.
fn quantize_block(data: &[f32], bits: u8) -> (Vec<u8>, f32) {
    let qm = qmax(bits);
    if qm == 0 || data.is_empty() {
        return (Vec::new(), 0.0);
    }
    let qm_f = qm as f32;

    // Find the maximum finite absolute value.
    let max_abs = data
        .iter()
        .filter(|v| v.is_finite())
        .fold(0.0f32, |acc, v| acc.max(v.abs()));

    let scale = if max_abs == 0.0 { 0.0 } else { max_abs / qm_f };
    let inv_scale = if scale == 0.0 { 0.0 } else { 1.0 / scale };

    let bits_u32 = bits as u32;
    let needed = (data.len() * bits as usize).div_ceil(8);
    let mut packed = Vec::with_capacity(needed);

    let mut acc: u64 = 0;
    let mut acc_bits: u32 = 0;

    for &v in data {
        let q = if v.is_finite() {
            (v * inv_scale).round() as i32
        } else {
            0
        }
        .clamp(-qm, qm);

        let u = (q + qm) as u32;
        acc |= (u as u64) << acc_bits;
        acc_bits += bits_u32;

        while acc_bits >= 8 {
            packed.push((acc & 0xFF) as u8);
            acc >>= 8;
            acc_bits -= 8;
        }
    }

    if acc_bits > 0 {
        packed.push((acc & 0xFF) as u8);
    }

    (packed, scale)
}

/// Dequantize packed bytes back to f32 using the given scale and bit width.
///
/// Writes up to `count` values into `out` and returns how many were written.
fn dequantize_block(packed: &[u8], scale: f32, bits: u8, count: usize, out: &mut [f32]) -> usize {
    let qm = qmax(bits);
    if qm == 0 || packed.is_empty() {
        return 0;
    }

    let bits_u32 = bits as u32;
    let mask = (1u64 << bits_u32) - 1;
    let limit = count.min(out.len());

    let mut acc: u64 = 0;
    let mut acc_bits: u32 = 0;
    let mut byte_idx: usize = 0;
    let mut written: usize = 0;

    while written < limit {
        while acc_bits < bits_u32 && byte_idx < packed.len() {
            acc |= (packed[byte_idx] as u64) << acc_bits;
            acc_bits += 8;
            byte_idx += 1;
        }
        if acc_bits < bits_u32 {
            break;
        }

        let u = (acc & mask) as i32;
        acc >>= bits_u32;
        acc_bits -= bits_u32;

        out[written] = (u - qm) as f32 * scale;
        written += 1;
    }

    written
}

/// Compute the CRC32 checksum over quantized payload concatenated with scale.
fn block_checksum(packed: &[u8], scale: f32) -> u32 {
    let scale_bytes = scale.to_le_bytes();
    let total = packed.len() + scale_bytes.len();
    let mut buf = Vec::with_capacity(total);
    buf.extend_from_slice(packed);
    buf.extend_from_slice(&scale_bytes);
    crc32(&buf)
}

// ---------------------------------------------------------------------------
// TieredStore
// ---------------------------------------------------------------------------

/// In-memory tiered storage engine for quantized tensor blocks.
///
/// Provides put/get with automatic quantization and dequantization,
/// per-block metadata tracking, access-pattern statistics, and
/// eviction to Tier0.
pub struct TieredStore {
    /// Nominal block size hint (bytes). Stored for reference; actual block
    /// sizes are determined by the data passed to [`put`].
    block_bytes: usize,

    /// Block metadata index keyed by [`BlockKey`].
    index: HashMap<BlockKey, BlockMeta>,

    /// Tier1 (hot, 8-bit) quantized data.
    tier1_data: HashMap<BlockKey, BlockData>,
    /// Tier2 (warm, 7-bit) quantized data.
    tier2_data: HashMap<BlockKey, BlockData>,
    /// Tier3 (cold, 3-bit) quantized data.
    tier3_data: HashMap<BlockKey, BlockData>,

    /// Keys present in each tier, for candidate-selection scans.
    tier1_keys: Vec<BlockKey>,
    tier2_keys: Vec<BlockKey>,
    tier3_keys: Vec<BlockKey>,
}

/// Smoothing constant for the exponential moving average of access rate.
const EMA_ALPHA: f32 = 0.1;

impl TieredStore {
    /// Create a new store with the given nominal block size (in bytes).
    pub fn new(block_bytes: usize) -> Self {
        Self {
            block_bytes,
            index: HashMap::new(),
            tier1_data: HashMap::new(),
            tier2_data: HashMap::new(),
            tier3_data: HashMap::new(),
            tier1_keys: Vec::new(),
            tier2_keys: Vec::new(),
            tier3_keys: Vec::new(),
        }
    }

    /// Nominal block size hint (bytes) configured at construction.
    #[inline]
    pub fn block_bytes(&self) -> usize {
        self.block_bytes
    }

    /// Quantize `data` at the bit width for `tier` and store the block.
    ///
    /// If a block with the same key already exists, it is replaced (the old
    /// data is removed from whatever tier it resided in).
    ///
    /// Returns [`StoreError::InvalidBlock`] if `tier` is [`Tier::Tier0`]
    /// (you cannot directly write to the evicted tier).
    pub fn put(
        &mut self,
        key: BlockKey,
        data: &[f32],
        tier: Tier,
        now: u64,
    ) -> Result<(), StoreError> {
        if tier == Tier::Tier0 {
            return Err(StoreError::InvalidBlock);
        }

        let bits = bits_for_tier(tier);
        let (packed, scale) = quantize_block(data, bits);
        let checksum = block_checksum(&packed, scale);

        // If the key already exists, remove old data first.
        if let Some(old_meta) = self.index.get(&key) {
            let old_tier = old_meta.tier;
            self.remove_data(old_tier, key);
            self.remove_from_bucket(old_tier, key);
        }

        let byte_count = packed.len() as u32;
        let block = BlockData {
            element_count: data.len() as u32,
            packed,
        };

        match tier {
            Tier::Tier1 => { self.tier1_data.insert(key, block); }
            Tier::Tier2 => { self.tier2_data.insert(key, block); }
            Tier::Tier3 => { self.tier3_data.insert(key, block); }
            Tier::Tier0 => unreachable!(),
        }
        self.add_to_bucket(tier, key);

        let meta = BlockMeta {
            key,
            dtype: DType::F32,
            tier,
            bits,
            scale,
            zero_point: 0,
            created_at: now,
            last_access_at: now,
            access_count: 1,
            ema_rate: 0.0,
            window: 1,
            checksum,
            reconstruct: ReconstructPolicy::None,
            tier_age: 0,
            lineage_parent: None,
            block_bytes: byte_count,
        };
        self.index.insert(key, meta);

        Ok(())
    }

    /// Dequantize the block identified by `key` into `out`.
    ///
    /// Returns the number of f32 elements written to `out`.
    ///
    /// # Errors
    ///
    /// - [`StoreError::TensorEvicted`] if the block resides in Tier0.
    /// - [`StoreError::BlockNotFound`] if no block exists for `key`.
    /// - [`StoreError::ChecksumMismatch`] if the stored checksum does not
    ///   match a freshly computed checksum of the payload.
    pub fn get(&self, key: BlockKey, out: &mut [f32]) -> Result<usize, StoreError> {
        let meta = self.index.get(&key).ok_or(StoreError::BlockNotFound)?;

        if meta.tier == Tier::Tier0 {
            return Err(StoreError::TensorEvicted);
        }

        let block = self
            .data_map(meta.tier)
            .and_then(|m| m.get(&key))
            .ok_or(StoreError::BlockNotFound)?;

        // Verify integrity.
        let actual_crc = block_checksum(&block.packed, meta.scale);
        if actual_crc != meta.checksum {
            return Err(StoreError::ChecksumMismatch);
        }

        let n = dequantize_block(
            &block.packed,
            meta.scale,
            meta.bits,
            block.element_count as usize,
            out,
        );
        Ok(n)
    }

    /// Update access statistics for `key` at tick `now`.
    ///
    /// Increments `access_count`, refreshes `last_access_at`, updates the
    /// sliding-window bitset, and recalculates the EMA access rate.
    /// Does nothing if the key is not present.
    pub fn touch(&mut self, key: BlockKey, now: u64) {
        if let Some(meta) = self.index.get_mut(&key) {
            let delta = now.saturating_sub(meta.last_access_at);

            // Update sliding-window bitset.
            if delta >= 64 {
                meta.window = 1;
            } else if delta > 0 {
                meta.window = (meta.window << delta) | 1;
            }
            // delta == 0: same tick, window unchanged but count still bumps.

            // Update EMA access rate.
            if delta > 0 {
                let instant_rate = 1.0 / delta as f32;
                meta.ema_rate = EMA_ALPHA * instant_rate + (1.0 - EMA_ALPHA) * meta.ema_rate;
            }

            meta.last_access_at = now;
            meta.access_count = meta.access_count.saturating_add(1);
        }
    }

    /// Return a reference to the metadata for `key`, if it exists.
    pub fn meta(&self, key: BlockKey) -> Option<&BlockMeta> {
        self.index.get(&key)
    }

    /// Total number of blocks tracked (including Tier0 evicted blocks).
    pub fn block_count(&self) -> usize {
        self.index.len()
    }

    /// Number of blocks currently in the given tier.
    pub fn tier_count(&self, tier: Tier) -> usize {
        match tier {
            Tier::Tier0 => self
                .index
                .values()
                .filter(|m| m.tier == Tier::Tier0)
                .count(),
            Tier::Tier1 => self.tier1_keys.len(),
            Tier::Tier2 => self.tier2_keys.len(),
            Tier::Tier3 => self.tier3_keys.len(),
        }
    }

    /// Total bytes of quantized data stored across all active tiers.
    pub fn total_bytes(&self) -> usize {
        let sum = |map: &HashMap<BlockKey, BlockData>| -> usize {
            map.values().map(|b| b.packed.len()).sum()
        };
        sum(&self.tier1_data) + sum(&self.tier2_data) + sum(&self.tier3_data)
    }

    /// Slice of block keys currently residing in the given tier.
    ///
    /// Returns an empty slice for [`Tier::Tier0`].
    pub fn blocks_in_tier(&self, tier: Tier) -> &[BlockKey] {
        match tier {
            Tier::Tier0 => &[],
            Tier::Tier1 => &self.tier1_keys,
            Tier::Tier2 => &self.tier2_keys,
            Tier::Tier3 => &self.tier3_keys,
        }
    }

    /// Evict a block to Tier0, removing its quantized payload.
    ///
    /// The block's metadata is preserved with the specified
    /// [`ReconstructPolicy`] so that higher-level code can decide how
    /// (or whether) to reconstruct the data on future reads.
    ///
    /// Returns [`StoreError::BlockNotFound`] if the key does not exist.
    pub fn evict(
        &mut self,
        key: BlockKey,
        policy: ReconstructPolicy,
    ) -> Result<(), StoreError> {
        let meta = self.index.get_mut(&key).ok_or(StoreError::BlockNotFound)?;
        let old_tier = meta.tier;

        if old_tier == Tier::Tier0 {
            // Already evicted; just update the policy.
            meta.reconstruct = policy;
            return Ok(());
        }

        // Mutate metadata before touching the data maps (avoids a second
        // lookup since we already have the mutable reference).
        meta.tier = Tier::Tier0;
        meta.reconstruct = policy;
        meta.tier_age = 0;
        meta.block_bytes = 0;
        meta.bits = 0;

        // Drop the mutable borrow so we can call helper methods.
        self.remove_data(old_tier, key);
        self.remove_from_bucket(old_tier, key);

        Ok(())
    }

    // -- private helpers ----------------------------------------------------

    /// Return a reference to the data map for the given tier.
    fn data_map(&self, tier: Tier) -> Option<&HashMap<BlockKey, BlockData>> {
        match tier {
            Tier::Tier0 => None,
            Tier::Tier1 => Some(&self.tier1_data),
            Tier::Tier2 => Some(&self.tier2_data),
            Tier::Tier3 => Some(&self.tier3_data),
        }
    }

    /// Remove raw data for `key` from the given tier's map.
    fn remove_data(&mut self, tier: Tier, key: BlockKey) {
        match tier {
            Tier::Tier1 => { self.tier1_data.remove(&key); }
            Tier::Tier2 => { self.tier2_data.remove(&key); }
            Tier::Tier3 => { self.tier3_data.remove(&key); }
            Tier::Tier0 => {}
        }
    }

    /// Remove `key` from the tier's candidate-selection bucket.
    fn remove_from_bucket(&mut self, tier: Tier, key: BlockKey) {
        let bucket = match tier {
            Tier::Tier1 => &mut self.tier1_keys,
            Tier::Tier2 => &mut self.tier2_keys,
            Tier::Tier3 => &mut self.tier3_keys,
            Tier::Tier0 => return,
        };
        if let Some(pos) = bucket.iter().position(|k| *k == key) {
            bucket.swap_remove(pos);
        }
    }

    /// Add `key` to the tier's candidate-selection bucket.
    fn add_to_bucket(&mut self, tier: Tier, key: BlockKey) {
        match tier {
            Tier::Tier1 => self.tier1_keys.push(key),
            Tier::Tier2 => self.tier2_keys.push(key),
            Tier::Tier3 => self.tier3_keys.push(key),
            Tier::Tier0 => {}
        }
    }
}

// ---------------------------------------------------------------------------
// Trait implementations for TieredStore
// ---------------------------------------------------------------------------

impl BlockIO for TieredStore {
    fn read_block(&self, tier: Tier, key: BlockKey, dst: &mut [u8]) -> Result<usize, StoreError> {
        let map = self.data_map(tier).ok_or(StoreError::BlockNotFound)?;
        let block = map.get(&key).ok_or(StoreError::BlockNotFound)?;
        let n = block.packed.len().min(dst.len());
        dst[..n].copy_from_slice(&block.packed[..n]);
        Ok(n)
    }

    fn write_block(&mut self, tier: Tier, key: BlockKey, src: &[u8]) -> Result<(), StoreError> {
        if tier == Tier::Tier0 {
            return Err(StoreError::InvalidBlock);
        }
        let block = BlockData {
            element_count: 0, // raw write; element count unknown
            packed: src.to_vec(),
        };
        match tier {
            Tier::Tier1 => { self.tier1_data.insert(key, block); }
            Tier::Tier2 => { self.tier2_data.insert(key, block); }
            Tier::Tier3 => { self.tier3_data.insert(key, block); }
            Tier::Tier0 => unreachable!(),
        }
        Ok(())
    }

    fn delete_block(&mut self, tier: Tier, key: BlockKey) -> Result<(), StoreError> {
        let removed = match tier {
            Tier::Tier1 => self.tier1_data.remove(&key).is_some(),
            Tier::Tier2 => self.tier2_data.remove(&key).is_some(),
            Tier::Tier3 => self.tier3_data.remove(&key).is_some(),
            Tier::Tier0 => false,
        };
        if removed {
            Ok(())
        } else {
            Err(StoreError::BlockNotFound)
        }
    }
}

impl MetaLog for TieredStore {
    fn append(&mut self, rec: &BlockMeta) -> Result<(), StoreError> {
        self.index.insert(rec.key, rec.clone());
        Ok(())
    }

    fn get(&self, key: BlockKey) -> Option<&BlockMeta> {
        self.index.get(&key)
    }

    fn iter(&self) -> Box<dyn Iterator<Item = &BlockMeta> + '_> {
        Box::new(self.index.values())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    fn make_key(tid: u128, idx: u32) -> BlockKey {
        BlockKey {
            tensor_id: tid,
            block_index: idx,
        }
    }

    // -- CRC32 -------------------------------------------------------------

    #[test]
    fn test_crc32_known_vector() {
        // The CRC32 of the ASCII string "123456789" is 0xCBF43926.
        let data = b"123456789";
        assert_eq!(crc32(data), 0xCBF4_3926);
    }

    #[test]
    fn test_crc32_empty() {
        assert_eq!(crc32(&[]), 0x0000_0000);
    }

    #[test]
    fn test_crc32_single_byte() {
        // CRC32 of [0x00] is 0xD202EF8D.
        assert_eq!(crc32(&[0x00]), 0xD202_EF8D);
    }

    // -- BlockKey hashing --------------------------------------------------

    #[test]
    fn test_block_key_equality() {
        let a = make_key(1, 0);
        let b = make_key(1, 0);
        let c = make_key(1, 1);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn test_block_key_hash_differs() {
        fn hash_of(k: &BlockKey) -> u64 {
            let mut h = DefaultHasher::new();
            k.hash(&mut h);
            h.finish()
        }
        let a = make_key(1, 0);
        let b = make_key(2, 0);
        let c = make_key(1, 1);
        // Different keys should (almost certainly) hash differently.
        assert_ne!(hash_of(&a), hash_of(&b));
        assert_ne!(hash_of(&a), hash_of(&c));
    }

    #[test]
    fn test_block_key_hash_stable() {
        fn hash_of(k: &BlockKey) -> u64 {
            let mut h = DefaultHasher::new();
            k.hash(&mut h);
            h.finish()
        }
        let a = make_key(42, 7);
        let b = make_key(42, 7);
        assert_eq!(hash_of(&a), hash_of(&b));
    }

    // -- qmax helper -------------------------------------------------------

    #[test]
    fn test_qmax_values() {
        assert_eq!(qmax(8), 127);
        assert_eq!(qmax(7), 63);
        assert_eq!(qmax(5), 15);
        assert_eq!(qmax(3), 3);
        assert_eq!(qmax(1), 0);
        assert_eq!(qmax(0), 0);
        assert_eq!(qmax(9), 0);
    }

    // -- Quantization roundtrip --------------------------------------------

    #[test]
    fn test_quantize_roundtrip_8bit() {
        let data: Vec<f32> = (0..128).map(|i| (i as f32 - 64.0) * 0.1).collect();
        let (packed, scale) = quantize_block(&data, 8);
        let mut out = vec![0.0f32; 128];
        let n = dequantize_block(&packed, scale, 8, 128, &mut out);
        assert_eq!(n, 128);
        for (i, (&orig, &dec)) in data.iter().zip(out.iter()).enumerate() {
            let err = (orig - dec).abs();
            let tol = if orig.abs() > 0.01 { orig.abs() * 0.02 } else { 0.1 };
            assert!(err < tol, "i={i} orig={orig} dec={dec} err={err}");
        }
    }

    #[test]
    fn test_quantize_roundtrip_3bit() {
        let data: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.5).collect();
        let (packed, scale) = quantize_block(&data, 3);
        let mut out = vec![0.0f32; 64];
        let n = dequantize_block(&packed, scale, 3, 64, &mut out);
        assert_eq!(n, 64);
        let max_val = data.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        for (&orig, &dec) in data.iter().zip(out.iter()) {
            let err = (orig - dec).abs();
            assert!(err < max_val * 0.35, "orig={orig} dec={dec} err={err}");
        }
    }

    #[test]
    fn test_quantize_zeros() {
        let data = vec![0.0f32; 64];
        let (packed, scale) = quantize_block(&data, 8);
        assert_eq!(scale, 0.0);
        let mut out = vec![1.0f32; 64];
        let n = dequantize_block(&packed, scale, 8, 64, &mut out);
        assert_eq!(n, 64);
        for &v in &out {
            assert_eq!(v, 0.0);
        }
    }

    // -- TieredStore put/get -----------------------------------------------

    #[test]
    fn test_store_put_get_roundtrip() {
        let mut store = TieredStore::new(4096);
        let key = make_key(1, 0);
        let data: Vec<f32> = (0..64).map(|i| i as f32 * 0.25).collect();

        store.put(key, &data, Tier::Tier1, 0).unwrap();

        let mut out = vec![0.0f32; 64];
        let n = store.get(key, &mut out).unwrap();
        assert_eq!(n, 64);

        for (i, (&orig, &dec)) in data.iter().zip(out.iter()).enumerate() {
            let err = (orig - dec).abs();
            let tol = if orig.abs() > 0.01 { orig.abs() * 0.02 } else { 0.15 };
            assert!(err < tol, "i={i} orig={orig} dec={dec} err={err}");
        }
    }

    #[test]
    fn test_store_put_tier3_roundtrip() {
        let mut store = TieredStore::new(4096);
        let key = make_key(10, 5);
        let data: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.5).collect();

        store.put(key, &data, Tier::Tier3, 100).unwrap();

        let meta = store.meta(key).unwrap();
        assert_eq!(meta.tier, Tier::Tier3);
        assert_eq!(meta.bits, 3);
        assert_eq!(meta.created_at, 100);

        let mut out = vec![0.0f32; 32];
        let n = store.get(key, &mut out).unwrap();
        assert_eq!(n, 32);

        let max_val = data.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        for (&orig, &dec) in data.iter().zip(out.iter()) {
            let err = (orig - dec).abs();
            assert!(err < max_val * 0.35, "orig={orig} dec={dec} err={err}");
        }
    }

    #[test]
    fn test_store_get_not_found() {
        let store = TieredStore::new(4096);
        let key = make_key(99, 0);
        let mut out = vec![0.0f32; 8];
        assert_eq!(store.get(key, &mut out), Err(StoreError::BlockNotFound));
    }

    #[test]
    fn test_store_put_tier0_rejected() {
        let mut store = TieredStore::new(4096);
        let key = make_key(1, 0);
        let data = vec![1.0f32; 8];
        assert_eq!(
            store.put(key, &data, Tier::Tier0, 0),
            Err(StoreError::InvalidBlock)
        );
    }

    // -- Eviction ----------------------------------------------------------

    #[test]
    fn test_eviction() {
        let mut store = TieredStore::new(4096);
        let key = make_key(1, 0);
        let data = vec![1.0f32; 64];

        store.put(key, &data, Tier::Tier1, 0).unwrap();
        assert_eq!(store.tier_count(Tier::Tier1), 1);
        assert!(store.total_bytes() > 0);

        store.evict(key, ReconstructPolicy::Delta).unwrap();

        let meta = store.meta(key).unwrap();
        assert_eq!(meta.tier, Tier::Tier0);
        assert_eq!(meta.reconstruct, ReconstructPolicy::Delta);
        assert_eq!(meta.block_bytes, 0);
        assert_eq!(meta.bits, 0);
        assert_eq!(meta.tier_age, 0);

        // Data is gone; read should fail with TensorEvicted.
        let mut out = vec![0.0f32; 64];
        assert_eq!(store.get(key, &mut out), Err(StoreError::TensorEvicted));

        // Tier1 should be empty; Tier0 count should be 1.
        assert_eq!(store.tier_count(Tier::Tier1), 0);
        assert_eq!(store.tier_count(Tier::Tier0), 1);

        // Block still exists in the index (metadata preserved).
        assert_eq!(store.block_count(), 1);
    }

    #[test]
    fn test_eviction_not_found() {
        let mut store = TieredStore::new(4096);
        let key = make_key(1, 0);
        assert_eq!(
            store.evict(key, ReconstructPolicy::None),
            Err(StoreError::BlockNotFound),
        );
    }

    #[test]
    fn test_eviction_idempotent() {
        let mut store = TieredStore::new(4096);
        let key = make_key(1, 0);
        store.put(key, &[1.0; 16], Tier::Tier2, 0).unwrap();

        store.evict(key, ReconstructPolicy::None).unwrap();
        // Evicting again should succeed and update the policy.
        store.evict(key, ReconstructPolicy::Factor).unwrap();

        let meta = store.meta(key).unwrap();
        assert_eq!(meta.reconstruct, ReconstructPolicy::Factor);
    }

    // -- Tier counts -------------------------------------------------------

    #[test]
    fn test_tier_counts() {
        let mut store = TieredStore::new(4096);
        let data = vec![1.0f32; 16];

        store.put(make_key(1, 0), &data, Tier::Tier1, 0).unwrap();
        store.put(make_key(2, 0), &data, Tier::Tier1, 0).unwrap();
        store.put(make_key(3, 0), &data, Tier::Tier2, 0).unwrap();
        store.put(make_key(4, 0), &data, Tier::Tier3, 0).unwrap();
        store.put(make_key(5, 0), &data, Tier::Tier3, 0).unwrap();
        store.put(make_key(6, 0), &data, Tier::Tier3, 0).unwrap();

        assert_eq!(store.block_count(), 6);
        assert_eq!(store.tier_count(Tier::Tier0), 0);
        assert_eq!(store.tier_count(Tier::Tier1), 2);
        assert_eq!(store.tier_count(Tier::Tier2), 1);
        assert_eq!(store.tier_count(Tier::Tier3), 3);

        assert_eq!(store.blocks_in_tier(Tier::Tier1).len(), 2);
        assert_eq!(store.blocks_in_tier(Tier::Tier0).len(), 0);
    }

    // -- Total bytes -------------------------------------------------------

    #[test]
    fn test_total_bytes() {
        let mut store = TieredStore::new(4096);
        assert_eq!(store.total_bytes(), 0);

        let data = vec![1.0f32; 64];
        store.put(make_key(1, 0), &data, Tier::Tier1, 0).unwrap();
        let bytes_after_one = store.total_bytes();
        assert!(bytes_after_one > 0);

        store.put(make_key(2, 0), &data, Tier::Tier2, 0).unwrap();
        assert!(store.total_bytes() > bytes_after_one);
    }

    #[test]
    fn test_total_bytes_decreases_on_evict() {
        let mut store = TieredStore::new(4096);
        let data = vec![1.0f32; 64];
        let key = make_key(1, 0);

        store.put(key, &data, Tier::Tier1, 0).unwrap();
        let before = store.total_bytes();

        store.evict(key, ReconstructPolicy::None).unwrap();
        assert_eq!(store.total_bytes(), before - before); // back to 0
    }

    // -- Touch / access stats ----------------------------------------------

    #[test]
    fn test_touch_updates_stats() {
        let mut store = TieredStore::new(4096);
        let key = make_key(1, 0);
        store.put(key, &[1.0; 16], Tier::Tier1, 0).unwrap();

        // Initial state after put.
        let meta = store.meta(key).unwrap();
        assert_eq!(meta.access_count, 1);
        assert_eq!(meta.last_access_at, 0);
        assert_eq!(meta.window, 1);

        // Touch at tick 5.
        store.touch(key, 5);
        let meta = store.meta(key).unwrap();
        assert_eq!(meta.access_count, 2);
        assert_eq!(meta.last_access_at, 5);
        // Window should have shifted left by 5 and gained bit 0.
        assert_eq!(meta.window, (1u64 << 5) | 1);
        assert!(meta.ema_rate > 0.0);

        // Touch at tick 5 again (same tick).
        store.touch(key, 5);
        let meta = store.meta(key).unwrap();
        assert_eq!(meta.access_count, 3);
        // Window unchanged on same-tick touch.
        assert_eq!(meta.window, (1u64 << 5) | 1);
    }

    #[test]
    fn test_touch_window_overflow() {
        let mut store = TieredStore::new(4096);
        let key = make_key(1, 0);
        store.put(key, &[1.0; 16], Tier::Tier1, 0).unwrap();

        // Touch after more than 64 ticks clears the window entirely.
        store.touch(key, 100);
        let meta = store.meta(key).unwrap();
        assert_eq!(meta.window, 1);
        assert_eq!(meta.last_access_at, 100);
    }

    #[test]
    fn test_touch_nonexistent_noop() {
        let mut store = TieredStore::new(4096);
        // Should not panic.
        store.touch(make_key(42, 0), 10);
    }

    // -- Overwrite ---------------------------------------------------------

    #[test]
    fn test_put_overwrite() {
        let mut store = TieredStore::new(4096);
        let key = make_key(1, 0);

        store.put(key, &[1.0; 16], Tier::Tier1, 0).unwrap();
        assert_eq!(store.tier_count(Tier::Tier1), 1);

        // Overwrite into a different tier.
        store.put(key, &[2.0; 16], Tier::Tier3, 10).unwrap();
        assert_eq!(store.block_count(), 1);
        assert_eq!(store.tier_count(Tier::Tier1), 0);
        assert_eq!(store.tier_count(Tier::Tier3), 1);

        let meta = store.meta(key).unwrap();
        assert_eq!(meta.tier, Tier::Tier3);
        assert_eq!(meta.created_at, 10);
    }

    // -- Checksum ----------------------------------------------------------

    #[test]
    fn test_checksum_stored_correctly() {
        let mut store = TieredStore::new(4096);
        let key = make_key(1, 0);
        let data: Vec<f32> = (0..32).map(|i| i as f32).collect();

        store.put(key, &data, Tier::Tier1, 0).unwrap();

        let meta = store.meta(key).unwrap();
        assert_ne!(meta.checksum, 0);

        // Manually verify the checksum matches.
        let (packed, scale) = quantize_block(&data, 8);
        let expected = block_checksum(&packed, scale);
        assert_eq!(meta.checksum, expected);
    }

    // -- BlockIO trait ------------------------------------------------------

    #[test]
    fn test_block_io_write_read() {
        let mut store = TieredStore::new(4096);
        let key = make_key(1, 0);
        let raw = vec![0xAA, 0xBB, 0xCC, 0xDD];

        store.write_block(Tier::Tier1, key, &raw).unwrap();

        let mut dst = vec![0u8; 8];
        let n = store.read_block(Tier::Tier1, key, &mut dst).unwrap();
        assert_eq!(n, 4);
        assert_eq!(&dst[..4], &raw);
    }

    #[test]
    fn test_block_io_delete() {
        let mut store = TieredStore::new(4096);
        let key = make_key(1, 0);
        store.write_block(Tier::Tier2, key, &[1, 2, 3]).unwrap();

        store.delete_block(Tier::Tier2, key).unwrap();

        let mut dst = vec![0u8; 4];
        assert_eq!(
            store.read_block(Tier::Tier2, key, &mut dst),
            Err(StoreError::BlockNotFound),
        );
    }

    #[test]
    fn test_block_io_write_tier0_rejected() {
        let mut store = TieredStore::new(4096);
        let key = make_key(1, 0);
        assert_eq!(
            store.write_block(Tier::Tier0, key, &[1]),
            Err(StoreError::InvalidBlock),
        );
    }

    // -- MetaLog trait ------------------------------------------------------

    #[test]
    fn test_meta_log_append_get() {
        let mut store = TieredStore::new(4096);
        let key = make_key(1, 0);
        let meta = BlockMeta {
            key,
            dtype: DType::F32,
            tier: Tier::Tier1,
            bits: 8,
            scale: 0.5,
            zero_point: 0,
            created_at: 42,
            last_access_at: 42,
            access_count: 1,
            ema_rate: 0.0,
            window: 1,
            checksum: 0,
            reconstruct: ReconstructPolicy::None,
            tier_age: 0,
            lineage_parent: None,
            block_bytes: 64,
        };

        MetaLog::append(&mut store, &meta).unwrap();
        let retrieved = MetaLog::get(&store, key).unwrap();
        assert_eq!(retrieved.key, key);
        assert_eq!(retrieved.created_at, 42);
    }

    #[test]
    fn test_meta_log_iter() {
        let mut store = TieredStore::new(4096);
        let data = vec![1.0f32; 8];

        store.put(make_key(1, 0), &data, Tier::Tier1, 0).unwrap();
        store.put(make_key(2, 0), &data, Tier::Tier2, 0).unwrap();
        store.put(make_key(3, 0), &data, Tier::Tier3, 0).unwrap();

        let entries: Vec<_> = MetaLog::iter(&store).collect();
        assert_eq!(entries.len(), 3);
    }

    // -- bits_for_tier -----------------------------------------------------

    #[test]
    fn test_bits_for_tier() {
        assert_eq!(bits_for_tier(Tier::Tier0), 0);
        assert_eq!(bits_for_tier(Tier::Tier1), 8);
        assert_eq!(bits_for_tier(Tier::Tier2), 7);
        assert_eq!(bits_for_tier(Tier::Tier3), 3);
    }

    // -- Tier enum ---------------------------------------------------------

    #[test]
    fn test_tier_repr() {
        assert_eq!(Tier::Tier0 as u8, 0);
        assert_eq!(Tier::Tier1 as u8, 1);
        assert_eq!(Tier::Tier2 as u8, 2);
        assert_eq!(Tier::Tier3 as u8, 3);
    }

    #[test]
    fn test_dtype_repr() {
        assert_eq!(DType::F32 as u8, 0);
        assert_eq!(DType::F16 as u8, 1);
        assert_eq!(DType::BF16 as u8, 2);
    }

    #[test]
    fn test_reconstruct_policy_repr() {
        assert_eq!(ReconstructPolicy::None as u8, 0);
        assert_eq!(ReconstructPolicy::Delta as u8, 1);
        assert_eq!(ReconstructPolicy::Factor as u8, 2);
    }

    // -- Integration: multi-block workflow ---------------------------------

    #[test]
    fn test_multi_block_workflow() {
        let mut store = TieredStore::new(4096);

        // Insert 10 blocks across tiers.
        for i in 0..10u32 {
            let key = make_key(1, i);
            let data: Vec<f32> = (0..32).map(|j| (i * 32 + j) as f32 * 0.1).collect();
            let tier = match i % 3 {
                0 => Tier::Tier1,
                1 => Tier::Tier2,
                _ => Tier::Tier3,
            };
            store.put(key, &data, tier, i as u64).unwrap();
        }

        assert_eq!(store.block_count(), 10);
        assert_eq!(store.tier_count(Tier::Tier1), 4); // 0,3,6,9
        assert_eq!(store.tier_count(Tier::Tier2), 3); // 1,4,7
        assert_eq!(store.tier_count(Tier::Tier3), 3); // 2,5,8

        // Touch some blocks.
        store.touch(make_key(1, 0), 20);
        store.touch(make_key(1, 5), 25);

        // Evict a cold block.
        store.evict(make_key(1, 8), ReconstructPolicy::Delta).unwrap();
        assert_eq!(store.tier_count(Tier::Tier3), 2);
        assert_eq!(store.tier_count(Tier::Tier0), 1);
        assert_eq!(store.block_count(), 10); // metadata preserved

        // Read back a hot block.
        let mut out = vec![0.0f32; 32];
        let n = store.get(make_key(1, 0), &mut out).unwrap();
        assert_eq!(n, 32);
    }
}
