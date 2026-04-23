//! RaBitQ-compressed cache. Wraps `ruvector_rabitq::RabitqPlusIndex`.
//!
//! Each `(backend, collection)` key maps to a cache entry that holds:
//! - the compressed index (RabitqPlusIndex with configurable rerank)
//! - the backend-reported generation at the time the entry was primed
//! - a parallel id map (u64 ids → internal u32 positions), because
//!   `RabitqPlusIndex` uses `usize` for its ID slot and the backend may
//!   use any u64 key space.
//!
//! Coherence model: on every search the router asks the backend for its
//! current generation and compares with the cache entry. On mismatch,
//! the entry is invalidated and re-primed. The check is opt-in per
//! search via the `consistency` parameter — tests default to
//! `Consistency::Fresh` (always re-check). Customers can choose
//! `Consistency::Eventual(ttl)` for higher QPS at cost of staleness.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use ruvector_rabitq::{AnnIndex, RabitqPlusIndex};

use crate::backend::{BackendId, CollectionId, PulledBatch};

/// How strictly the cache checks freshness before answering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Consistency {
    /// Consult the backend's generation counter on every search.
    /// Default in tests. Adds one backend round-trip per search, but
    /// guarantees freshness up to the backend's own coherence resolution.
    #[default]
    Fresh,
    /// Trust the cache for up to `ttl_ms` milliseconds between checks.
    /// Higher QPS; backend updates may be ignored for up to ttl.
    Eventual { ttl_ms: u64 },
}

/// Simple struct returned by `CacheStats::snapshot`.
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub primes: u64,
    pub invalidations: u64,
}

/// Key into the cache. Two backends can use the same collection name
/// independently.
pub type CacheKey = (BackendId, CollectionId);

struct CacheEntry {
    index: RabitqPlusIndex,
    dim: usize,
    generation: u64,
    last_checked: Instant,
    /// internal-position → external id.
    pos_to_id: Vec<u64>,
}

/// Thread-safe cache of RaBitQ-compressed per-collection indexes.
pub struct VectorCache {
    inner: Arc<Mutex<CacheState>>,
    rerank_factor: usize,
    rotation_seed: u64,
}

struct CacheState {
    entries: HashMap<CacheKey, CacheEntry>,
    stats: CacheStats,
}

impl VectorCache {
    /// Create a new cache. `rerank_factor` is passed straight to
    /// `RabitqPlusIndex`; 20 is the value from BENCHMARK.md that holds
    /// 100 % recall@10 at n ≥ 50 k on clustered Gaussian.
    pub fn new(rerank_factor: usize, rotation_seed: u64) -> Self {
        Self {
            inner: Arc::new(Mutex::new(CacheState {
                entries: HashMap::new(),
                stats: CacheStats::default(),
            })),
            rerank_factor,
            rotation_seed,
        }
    }

    pub fn stats(&self) -> CacheStats {
        self.inner.lock().unwrap().stats.clone()
    }

    /// Compress a pulled batch into a RaBitQ index and store under the
    /// given key. Overwrites any existing entry (used for invalidation +
    /// re-prime).
    pub fn prime(&self, key: CacheKey, batch: PulledBatch) -> crate::Result<()> {
        let mut idx = RabitqPlusIndex::new(batch.dim, self.rotation_seed, self.rerank_factor);
        // We intentionally don't trust the backend's u64 id to fit in
        // RabitqPlusIndex's usize slot — use the array position as the
        // internal id, store the mapping separately.
        let mut pos_to_id = Vec::with_capacity(batch.ids.len());
        for (pos, v) in batch.vectors.into_iter().enumerate() {
            idx.add(pos, v)?;
            pos_to_id.push(batch.ids[pos]);
        }
        let entry = CacheEntry {
            index: idx,
            dim: batch.dim,
            generation: batch.generation,
            last_checked: Instant::now(),
            pos_to_id,
        };
        let mut inner = self.inner.lock().unwrap();
        inner.entries.insert(key, entry);
        inner.stats.primes += 1;
        Ok(())
    }

    /// Drop the cache entry for a given key (used by explicit invalidation).
    pub fn invalidate(&self, key: &CacheKey) {
        let mut inner = self.inner.lock().unwrap();
        if inner.entries.remove(key).is_some() {
            inner.stats.invalidations += 1;
        }
    }

    /// Introspection — is there a live entry for this key?
    pub fn has(&self, key: &CacheKey) -> bool {
        self.inner.lock().unwrap().entries.contains_key(key)
    }

    /// Introspection — what generation does the cache currently hold?
    pub fn generation(&self, key: &CacheKey) -> Option<u64> {
        self.inner
            .lock()
            .unwrap()
            .entries
            .get(key)
            .map(|e| e.generation)
    }

    /// Introspection — what dim does the cache hold for this key?
    pub fn dim(&self, key: &CacheKey) -> Option<usize> {
        self.inner.lock().unwrap().entries.get(key).map(|e| e.dim)
    }

    /// Internal helper: record a hit / miss.
    pub(crate) fn mark_hit(&self) {
        self.inner.lock().unwrap().stats.hits += 1;
    }
    pub(crate) fn mark_miss(&self) {
        self.inner.lock().unwrap().stats.misses += 1;
    }

    /// Run the search. Returns (id, score) pairs. Must only be called
    /// when the entry is known-fresh by the router.
    pub fn search_cached(
        &self,
        key: &CacheKey,
        query: &[f32],
        k: usize,
    ) -> crate::Result<Vec<(u64, f32)>> {
        let inner = self.inner.lock().unwrap();
        let entry =
            inner
                .entries
                .get(key)
                .ok_or_else(|| crate::RuLakeError::UnknownCollection {
                    backend: key.0.clone(),
                    collection: key.1.clone(),
                })?;
        if query.len() != entry.dim {
            return Err(crate::RuLakeError::DimensionMismatch {
                expected: entry.dim,
                actual: query.len(),
            });
        }
        let hits = entry.index.search(query, k)?;
        Ok(hits
            .into_iter()
            .map(|r| (entry.pos_to_id[r.id], r.score))
            .collect())
    }

    /// Mark the entry as just-checked-fresh at this instant. Used by the
    /// router when `Consistency::Eventual` applies.
    pub fn touch(&self, key: &CacheKey) {
        if let Some(e) = self.inner.lock().unwrap().entries.get_mut(key) {
            e.last_checked = Instant::now();
        }
    }

    /// Can the cache answer the next search without re-checking the
    /// backend? Only true under `Consistency::Eventual` when the TTL has
    /// not elapsed since the last check.
    pub fn can_skip_check(&self, key: &CacheKey, consistency: Consistency) -> bool {
        match consistency {
            Consistency::Fresh => false,
            Consistency::Eventual { ttl_ms } => {
                let inner = self.inner.lock().unwrap();
                match inner.entries.get(key) {
                    Some(e) => e.last_checked.elapsed().as_millis() < ttl_ms as u128,
                    None => false,
                }
            }
        }
    }
}
