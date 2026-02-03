//! BitNet b1.58 Ternary Quantization for RuvLLM
//!
//! This module implements Microsoft Research's BitNet b1.58 ternary weight quantization
//! for the Craftsman Ultra 30b 1bit model. It provides post-training quantization (PTQ)
//! of FP16 weights to ternary {-1, 0, +1} using absmean quantization.
//!
//! ## Overview
//!
//! BitNet b1.58 enables multiplication-free inference by quantizing weights to three values:
//! -1, 0, +1. This reduces memory footprint to ~2 bits per weight and eliminates floating-point
//! multiplication in matrix operations.
//!
//! ## Key Components
//!
//! - [`TernaryTensor`]: Container for ternary weights with 2-bit packing
//! - [`quantize_tensor`]: Convert FP32 weights to ternary using absmean algorithm
//! - [`dequantize_bitnet_t158`]: Convert packed ternary back to FP32 for validation
//! - [`PtBitnetConfig`]: Configuration for post-training quantization
//!
//! ## Example
//!
//! ```rust,ignore
//! use ruvllm::bitnet::{quantize_tensor, PtBitnetConfig};
//!
//! // Configure quantization
//! let config = PtBitnetConfig {
//!     block_size: 256,
//!     optimize_scales: true,
//!     ..Default::default()
//! };
//!
//! // Quantize a weight tensor
//! let fp32_weights = vec![0.5, -0.3, 0.0, 0.8, /* ... */];
//! let ternary = quantize_tensor(&fp32_weights, (128, 256), &config)?;
//!
//! println!("Sparsity: {:.2}%", ternary.sparsity() * 100.0);
//! println!("Memory: {} bytes", ternary.memory_bytes());
//! ```
//!
//! ## Architecture Details
//!
//! From ADR-017 (AD-1, AD-5, AD-18):
//!
//! - **Absmean quantization**: `W_ternary = RoundClip(W / (mean(|W|) + ε), -1, 1)`
//! - **2-bit packing**: 00=-1, 01=0, 10=+1 (4 values per byte)
//! - **Block size**: 256 elements per scale factor
//! - **Storage**: 66 bytes per block (64 bytes ternary + 2 bytes FP16 scale)
//! - **Compression**: 2.06 bits/weight (30B model → ~7.7 GB)

pub mod dequantize;
pub mod quantizer;
pub mod ternary_tensor;

pub use dequantize::dequantize_bitnet_t158;
pub use quantizer::{
    absmean_ternary, quantize_tensor, LayerMask, Precision, PtBitnetConfig, TernaryFormat,
};
pub use ternary_tensor::{pack_ternary, unpack_ternary, TernaryTensor};
