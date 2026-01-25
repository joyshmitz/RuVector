//! Performance Optimizations for j-Tree + BMSSP Implementation
//!
//! This module implements the SOTA optimizations from ADR-002-addendum-sota-optimizations.md:
//!
//! 1. **Degree-based presparse (DSpar)**: 5.9x speedup via effective resistance approximation
//! 2. **LRU Cache**: Path distance caching with prefetch optimization
//! 3. **SIMD Operations**: Vectorized distance array computations
//! 4. **Pool Allocators**: Memory-efficient allocations with lazy deallocation
//! 5. **Parallel Updates**: Rayon-based parallel level updates with work-stealing
//! 6. **WASM Optimization**: Batch operations and TypedArray transfers
//!
//! Target: Combined 10x speedup over naive implementation.

pub mod dspar;
pub mod cache;
pub mod simd_distance;
pub mod pool;
pub mod parallel;
pub mod wasm_batch;
pub mod benchmark;

// Re-exports
pub use dspar::{DegreePresparse, PresparseConfig, PresparseResult, PresparseStats};
pub use cache::{PathDistanceCache, CacheConfig, CacheStats, PrefetchHint};
pub use simd_distance::{SimdDistanceOps, DistanceArray};
pub use pool::{LevelPool, PoolConfig, LazyLevel, PoolStats};
pub use parallel::{ParallelLevelUpdater, ParallelConfig, WorkStealingScheduler};
pub use wasm_batch::{WasmBatchOps, BatchConfig, TypedArrayTransfer};
pub use benchmark::{BenchmarkSuite, BenchmarkResult, OptimizationBenchmark};
