//! Swarm Intelligence for edge-net P2P Network
//!
//! This module provides swarm coordination mechanisms for self-organizing
//! task allocation and emergent network behavior.
//!
//! ## Components
//!
//! - **Stigmergy**: Digital pheromones for self-organizing task allocation
//!   - Deposit/decay mechanics with anti-sybil protection
//!   - P2P trail synchronization via gossip
//!   - Emergent specialization through gradient following
//!   - Self-healing through pheromone evaporation
//!
//! ## Future Components (planned)
//!
//! - **Consensus**: Entropy-based distributed decision making
//!   - Belief propagation
//!   - Entropy minimization for convergence
//!   - Byzantine fault tolerance
//!
//! - **Collective**: Network-wide memory formation
//!   - Hippocampal-inspired consolidation
//!   - RAC-based pattern sharing
//!   - Quality-gated storage

pub mod stigmergy;

// Re-export stigmergy types
pub use stigmergy::{
    PeerId, PheromoneDeposit, PheromoneState, PheromoneTrail, RingBuffer, Stigmergy,
    StigmergyStats, WasmStigmergy,
};
