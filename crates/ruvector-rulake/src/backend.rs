//! Backend-adapter trait. Every supported data lake (Parquet-on-S3,
//! BigQuery, Snowflake, Delta, Iceberg, …) implements this.
//!
//! ## The minimum surface
//!
//! - `id()` — a stable string identifier unique per-backend instance.
//! - `list_collections()` — what vector collections live in this backend.
//! - `pull_vectors(collection)` — stream all vectors in the collection.
//!   Called on cache miss / coherence bump.
//! - `generation(collection)` — an opaque coherence token (Parquet file
//!   mtime, Iceberg snapshot id, BQ `last_modified_time`, …). Used to
//!   decide whether the cache for this collection is still fresh.
//! - `supports_pushdown()` — optional; defaults to `false`. When `true`,
//!   the router may choose to push top-k ANN search into the backend
//!   instead of pulling all vectors. v1 does not actually call push-down;
//!   the flag is the forward-compatibility hook.
//!
//! `LocalBackend` is the in-memory reference implementation — used by
//! tests, demos, and the "does the federation path round-trip correctly"
//! smoke.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use crate::error::{Result, RuLakeError};

/// Stable identifier for a registered backend.
pub type BackendId = String;

/// Collection name inside a given backend — globally unique only in
/// conjunction with its `BackendId`.
pub type CollectionId = String;

/// One pull from a backend. Vectors are returned by value — we assume
/// the caller (the cache) is the only consumer and will compress them
/// into 1-bit codes immediately.
#[derive(Debug, Clone)]
pub struct PulledBatch {
    /// Collection this batch belongs to.
    pub collection: CollectionId,
    /// Parallel arrays: `ids[i]` and `vectors[i]` describe vector i.
    pub ids: Vec<u64>,
    /// Each vector must have length `dim`.
    pub vectors: Vec<Vec<f32>>,
    /// Dimensionality reported by the backend.
    pub dim: usize,
    /// Coherence token — when this bumps the cache is stale.
    pub generation: u64,
}

pub trait BackendAdapter: Send + Sync {
    fn id(&self) -> &str;

    fn list_collections(&self) -> Result<Vec<CollectionId>>;

    fn pull_vectors(&self, collection: &str) -> Result<PulledBatch>;

    fn generation(&self, collection: &str) -> Result<u64>;

    fn supports_pushdown(&self) -> bool {
        false
    }
}

// ────────────────────────────────────────────────────────────────────
// LocalBackend — in-memory reference impl
// ────────────────────────────────────────────────────────────────────

/// In-memory backend. Useful as a demo, as the unit-test substrate, and
/// as an example for real-backend implementers (ParquetBackend,
/// BigQueryBackend, …). Thread-safe: the inner collections table is
/// guarded by an `RwLock` so the backend can be shared across threads.
#[derive(Clone)]
pub struct LocalBackend {
    id: String,
    inner: Arc<RwLock<LocalState>>,
}

struct LocalState {
    collections: HashMap<CollectionId, LocalCollection>,
}

struct LocalCollection {
    dim: usize,
    ids: Vec<u64>,
    vectors: Vec<Vec<f32>>,
    generation: u64,
}

impl LocalBackend {
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            inner: Arc::new(RwLock::new(LocalState {
                collections: HashMap::new(),
            })),
        }
    }

    /// Insert a collection wholesale. Bumps the generation so any cache
    /// watching this collection sees it as stale on the next check.
    pub fn put_collection(
        &self,
        name: impl Into<String>,
        dim: usize,
        ids: Vec<u64>,
        vectors: Vec<Vec<f32>>,
    ) -> Result<()> {
        if ids.len() != vectors.len() {
            return Err(RuLakeError::InvalidParameter(format!(
                "put_collection: ids.len={} != vectors.len={}",
                ids.len(),
                vectors.len()
            )));
        }
        for v in &vectors {
            if v.len() != dim {
                return Err(RuLakeError::DimensionMismatch {
                    expected: dim,
                    actual: v.len(),
                });
            }
        }
        let mut inner = self.inner.write().unwrap();
        let entry = inner
            .collections
            .entry(name.into())
            .or_insert(LocalCollection {
                dim,
                ids: Vec::new(),
                vectors: Vec::new(),
                generation: 0,
            });
        entry.dim = dim;
        entry.ids = ids;
        entry.vectors = vectors;
        entry.generation = entry.generation.wrapping_add(1);
        Ok(())
    }

    /// Append a single vector. Bumps generation.
    pub fn append(&self, collection: impl Into<String>, id: u64, vector: Vec<f32>) -> Result<()> {
        let mut inner = self.inner.write().unwrap();
        let name = collection.into();
        let entry =
            inner
                .collections
                .get_mut(&name)
                .ok_or_else(|| RuLakeError::UnknownCollection {
                    backend: self.id.clone(),
                    collection: name.clone(),
                })?;
        if entry.dim == 0 {
            entry.dim = vector.len();
        }
        if vector.len() != entry.dim {
            return Err(RuLakeError::DimensionMismatch {
                expected: entry.dim,
                actual: vector.len(),
            });
        }
        entry.ids.push(id);
        entry.vectors.push(vector);
        entry.generation = entry.generation.wrapping_add(1);
        Ok(())
    }
}

impl BackendAdapter for LocalBackend {
    fn id(&self) -> &str {
        &self.id
    }

    fn list_collections(&self) -> Result<Vec<CollectionId>> {
        Ok(self
            .inner
            .read()
            .unwrap()
            .collections
            .keys()
            .cloned()
            .collect())
    }

    fn pull_vectors(&self, collection: &str) -> Result<PulledBatch> {
        let inner = self.inner.read().unwrap();
        let c =
            inner
                .collections
                .get(collection)
                .ok_or_else(|| RuLakeError::UnknownCollection {
                    backend: self.id.clone(),
                    collection: collection.to_string(),
                })?;
        Ok(PulledBatch {
            collection: collection.to_string(),
            ids: c.ids.clone(),
            vectors: c.vectors.clone(),
            dim: c.dim,
            generation: c.generation,
        })
    }

    fn generation(&self, collection: &str) -> Result<u64> {
        let inner = self.inner.read().unwrap();
        let c =
            inner
                .collections
                .get(collection)
                .ok_or_else(|| RuLakeError::UnknownCollection {
                    backend: self.id.clone(),
                    collection: collection.to_string(),
                })?;
        Ok(c.generation)
    }
}
