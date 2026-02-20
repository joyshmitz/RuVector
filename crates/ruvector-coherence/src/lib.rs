//! Coherence measurement proxies for comparing attention mechanisms.
//!
//! This crate provides metrics, comparison utilities, quality guardrails,
//! and batched evaluation tools for measuring how different attention
//! mechanisms (e.g., baseline vs. gated) affect output coherence.

pub mod batch;
pub mod comparison;
pub mod metrics;
pub mod quality;

pub use batch::{evaluate_batch, BatchResult};
pub use comparison::{
    compare_attention_masks, edge_flip_count, jaccard_similarity, ComparisonResult,
};
pub use metrics::{contradiction_rate, delta_behavior, entailment_consistency, DeltaMetric};
pub use quality::{cosine_similarity, l2_distance, quality_check, QualityResult};
