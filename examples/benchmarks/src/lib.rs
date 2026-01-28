//! RuVector Benchmarks Library
//!
//! Comprehensive benchmarking suite for:
//! - Temporal reasoning (TimePuzzles-style constraint inference)
//! - Vector index operations (IVF, coherence-gated search)
//! - Swarm controller regret tracking
//! - Intelligence metrics and cognitive capability assessment
//! - Adaptive learning with ReasoningBank trajectory tracking
//!
//! Based on research from:
//! - TimePuzzles benchmark (arXiv:2601.07148)
//! - Sublinear regret in multi-agent control
//! - Tool-augmented iterative temporal reasoning
//! - Cognitive capability assessment frameworks
//! - lean-agentic type theory for verified reasoning

pub mod intelligence_metrics;
pub mod logging;
pub mod reasoning_bank;
pub mod swarm_regret;
pub mod temporal;
pub mod timepuzzles;
pub mod vector_index;

pub use intelligence_metrics::*;
pub use logging::*;
pub use reasoning_bank::*;
pub use swarm_regret::*;
pub use temporal::*;
pub use timepuzzles::*;
pub use vector_index::*;
