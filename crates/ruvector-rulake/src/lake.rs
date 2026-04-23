//! The public `RuLake` entry point — registers backends, routes searches.
//!
//! v1 fans out across registered backends, runs RaBitQ-cache searches,
//! and merges the results by score. v2 will push-down to backends that
//! support native vector ops.

use std::collections::HashMap;
use std::sync::Arc;

use crate::backend::{BackendAdapter, BackendId};
use crate::cache::{CacheKey, Consistency, VectorCache};
use crate::error::{Result, RuLakeError};

/// Result from a search — the external id and its estimated L2² score.
/// Includes the backend that produced the hit so callers can audit.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchResult {
    pub backend: BackendId,
    pub collection: String,
    pub id: u64,
    pub score: f32,
}

/// ruLake entry point. Cheap to clone (everything is behind `Arc`).
#[derive(Clone)]
pub struct RuLake {
    backends: Arc<std::sync::RwLock<HashMap<BackendId, Arc<dyn BackendAdapter>>>>,
    cache: Arc<VectorCache>,
    consistency: Consistency,
}

impl RuLake {
    /// Build a fresh ruLake. `rerank_factor` controls the RaBitQ cache
    /// precision (20 → 100% recall@10 on clustered D=128 at n ≥ 50k per
    /// `ruvector-rabitq::BENCHMARK.md`). `rotation_seed` is shared across
    /// all cached collections so the compression is deterministic
    /// (important for the reproducibility + witness story).
    pub fn new(rerank_factor: usize, rotation_seed: u64) -> Self {
        Self {
            backends: Arc::new(std::sync::RwLock::new(HashMap::new())),
            cache: Arc::new(VectorCache::new(rerank_factor, rotation_seed)),
            consistency: Consistency::default(),
        }
    }

    /// Set the cache consistency mode. Defaults to `Consistency::Fresh`.
    pub fn with_consistency(mut self, c: Consistency) -> Self {
        self.consistency = c;
        self
    }

    /// Register a backend under its `id()`. Returns an error if a backend
    /// with the same id already exists.
    pub fn register_backend(&self, backend: Arc<dyn BackendAdapter>) -> Result<()> {
        let mut map = self.backends.write().unwrap();
        let id = backend.id().to_string();
        if map.contains_key(&id) {
            return Err(RuLakeError::InvalidParameter(format!(
                "backend {id} already registered"
            )));
        }
        map.insert(id, backend);
        Ok(())
    }

    pub fn backend_ids(&self) -> Vec<BackendId> {
        self.backends.read().unwrap().keys().cloned().collect()
    }

    /// Access the cache stats for diagnostics / benchmarking.
    pub fn cache_stats(&self) -> crate::CacheStats {
        self.cache.stats()
    }

    /// Search a single (backend, collection) pair. Handles cache
    /// miss / staleness transparently.
    pub fn search_one(
        &self,
        backend: &str,
        collection: &str,
        query: &[f32],
        k: usize,
    ) -> Result<Vec<SearchResult>> {
        let key: CacheKey = (backend.to_string(), collection.to_string());
        self.ensure_fresh(&key)?;
        let hits = self.cache.search_cached(&key, query, k)?;
        Ok(hits
            .into_iter()
            .map(|(id, score)| SearchResult {
                backend: backend.to_string(),
                collection: collection.to_string(),
                id,
                score,
            })
            .collect())
    }

    /// Federated search: fan out to every `(backend, collection)` pair
    /// in `targets`, merge by score, return global top-k.
    pub fn search_federated(
        &self,
        targets: &[(&str, &str)],
        query: &[f32],
        k: usize,
    ) -> Result<Vec<SearchResult>> {
        let mut merged: Vec<SearchResult> = Vec::with_capacity(targets.len() * k);
        for (backend, collection) in targets {
            let hits = self.search_one(backend, collection, query, k)?;
            merged.extend(hits);
        }
        // Ascending by score (L2²) — smaller = closer.
        merged.sort_by(|a, b| a.score.total_cmp(&b.score));
        merged.truncate(k);
        Ok(merged)
    }

    /// Coherence check: consult the backend's generation, re-prime the
    /// cache if stale or absent. Respects `self.consistency`.
    fn ensure_fresh(&self, key: &CacheKey) -> Result<()> {
        // Fast path: Eventual mode + within-TTL → skip check.
        if self.cache.can_skip_check(key, self.consistency) {
            self.cache.mark_hit();
            return Ok(());
        }

        let backend = self.get_backend(&key.0)?;
        let current_gen = backend.generation(&key.1)?;
        let cache_gen = self.cache.generation(key);

        match cache_gen {
            Some(cg) if cg == current_gen => {
                // Hit — just update the last-checked timestamp so
                // Eventual-mode TTLs count from here.
                self.cache.mark_hit();
                self.cache.touch(key);
                Ok(())
            }
            _ => {
                // Miss or stale — pull + prime.
                self.cache.mark_miss();
                let batch = backend.pull_vectors(&key.1)?;
                self.cache.prime(key.clone(), batch)?;
                Ok(())
            }
        }
    }

    fn get_backend(&self, id: &str) -> Result<Arc<dyn BackendAdapter>> {
        self.backends
            .read()
            .unwrap()
            .get(id)
            .cloned()
            .ok_or_else(|| RuLakeError::UnknownBackend(id.to_string()))
    }
}
