//! Modern Hopfield Networks
//!
//! This module implements the modern Hopfield network formulation from
//! Ramsauer et al. (2020), which provides exponential storage capacity
//! and is mathematically equivalent to transformer attention.
//!
//! ## Components
//!
//! - [`ModernHopfield`]: Main network structure
//! - [`retrieval`]: Softmax-weighted retrieval implementation
//! - [`capacity`]: Capacity calculations and β tuning

mod network;
mod retrieval;
mod capacity;

pub use network::ModernHopfield;
pub use retrieval::{compute_attention, softmax};
pub use capacity::{theoretical_capacity, optimal_beta};

#[cfg(test)]
mod tests;
