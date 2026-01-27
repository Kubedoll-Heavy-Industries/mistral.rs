//! Weight source abstraction for unified model loading.
//!
//! This module provides traits that abstract over how weights are loaded,
//! allowing model infrastructure to work identically regardless of whether
//! weights come from GGUF, safetensors, or other sources.
//!
//! # Trait Hierarchy
//!
//! ```text
//! WeightSource          - Base trait for all weight loading
//!     │
//!     └── MoEWeightSource   - MoE-specific extensions (expert loading, backend selection)
//! ```
//!
//! # Design Principles
//!
//! 1. **Format is a loader detail**: The serialization format (GGUF vs safetensors)
//!    should not affect runtime execution strategy.
//!
//! 2. **Backend selection by tensor properties**: The execution backend (Fused/Fast/Slow)
//!    is chosen based on device + dtype + quantization, not file format.
//!
//! 3. **Unified type output**: All weight sources produce `Arc<dyn QuantMethod>`,
//!    enabling identical downstream handling.
//!
//! 4. **Refactor, don't wrap**: Implementations are clean new code, not wrappers
//!    around legacy abstractions.

use candle_core::{DType, Device, Result};
use mistralrs_quant::QuantMethod;
use std::sync::Arc;

// ============================================================================
// Base WeightSource Trait
// ============================================================================

/// Base trait for loading model weights from any source.
///
/// This trait provides a unified interface for loading weights regardless of
/// the underlying storage format (GGUF, safetensors, in-memory, etc.).
///
/// # Implementors
///
/// - `GgufWeightSource` - Loads from GGUF quantized model files
/// - `SafetensorsWeightSource` - Loads from safetensors files
///
/// # Example
///
/// ```ignore
/// fn load_attention<W: WeightSource>(weights: &W, prefix: &str) -> Result<Attention> {
///     let q_proj = weights.get_weight(&format!("{prefix}.q_proj.weight"))?;
///     let k_proj = weights.get_weight(&format!("{prefix}.k_proj.weight"))?;
///     let v_proj = weights.get_weight(&format!("{prefix}.v_proj.weight"))?;
///     let o_proj = weights.get_weight(&format!("{prefix}.o_proj.weight"))?;
///     // ...
/// }
/// ```
pub trait WeightSource: Send + Sync {
    /// Load a weight tensor by name.
    ///
    /// Returns the weight wrapped as `Arc<dyn QuantMethod>` for unified handling.
    /// The implementation handles format-specific loading and wrapping.
    ///
    /// # Arguments
    ///
    /// * `name` - Full tensor path (e.g., "model.layers.0.self_attn.q_proj.weight")
    fn get_weight(&self, name: &str) -> Result<Arc<dyn QuantMethod>>;

    /// Check if a tensor exists in this source.
    ///
    /// Used for format detection (e.g., detecting stacked vs per-expert weights).
    fn contains_tensor(&self, name: &str) -> bool;

    /// Get the target device for loaded tensors.
    fn device(&self) -> &Device;

    /// Get the default dtype for loaded tensors.
    fn dtype(&self) -> DType;

    /// Get a raw tensor without QuantMethod wrapping.
    ///
    /// Some use cases need the raw tensor (e.g., embeddings, norms).
    /// Default implementation extracts from QuantMethod if possible.
    fn get_raw_tensor(&self, name: &str) -> Result<candle_core::Tensor> {
        // Default: load via get_weight and try to extract
        // Implementations should override for efficiency
        let qmethod = self.get_weight(name)?;
        qmethod
            .unquant_weight_bias()
            .map(|(w, _)| w)
            .ok_or_else(|| {
                candle_core::Error::Msg(format!(
                    "Cannot extract raw tensor from quantized weight: {name}"
                ))
            })
    }
}

// ============================================================================
// MoE-Specific Extension
// ============================================================================

/// Properties of loaded weights that inform MoE backend selection.
///
/// These properties describe the characteristics of the weights
/// after loading, independent of the source format.
#[derive(Debug, Clone)]
pub struct QuantProperties {
    /// Quantization format (None for F16/BF16/F32).
    ///
    /// When set, indicates the weights are quantized (e.g., Q4_K, Q8_0).
    pub quant_format: Option<candle_core::quantized::GgmlDType>,

    /// Whether all experts have the same quantization format.
    ///
    /// When true, optimized batched operations may be possible.
    /// When false, must fall back to per-expert handling.
    pub uniform_quantization: bool,

    /// Whether weights support indexed MoE forward kernels.
    ///
    /// CUDA has optimized `indexed_moe_forward_*` kernels for certain
    /// quantization formats. This flag indicates compatibility.
    pub supports_indexed_forward: bool,

    /// Whether weights are pre-quantized (GGUF) vs runtime quantized (safetensors+ISQ).
    ///
    /// Pre-quantized weights cannot use fused CUDA kernels that expect raw tensors.
    pub is_prequantized: bool,
}

impl Default for QuantProperties {
    fn default() -> Self {
        Self {
            quant_format: None,
            uniform_quantization: true,
            supports_indexed_forward: true,
            is_prequantized: false,
        }
    }
}

impl QuantProperties {
    /// Create properties for unquantized weights (F16/BF16/F32).
    pub fn unquantized() -> Self {
        Self::default()
    }

    /// Create properties for quantized GGUF weights.
    pub fn gguf(dtype: candle_core::quantized::GgmlDType) -> Self {
        Self {
            quant_format: Some(dtype),
            uniform_quantization: true,
            is_prequantized: true,
            // These GGUF quantization formats have optimized indexed_moe_forward kernels
            supports_indexed_forward: matches!(
                dtype,
                candle_core::quantized::GgmlDType::Q2K
                    | candle_core::quantized::GgmlDType::Q3K
                    | candle_core::quantized::GgmlDType::Q4K
                    | candle_core::quantized::GgmlDType::Q5K
                    | candle_core::quantized::GgmlDType::Q6K
                    | candle_core::quantized::GgmlDType::Q8_0
            ),
        }
    }

    /// Create properties for ISQ (in-situ quantization).
    pub fn isq() -> Self {
        Self {
            quant_format: None, // ISQ happens after loading
            uniform_quantization: true,
            supports_indexed_forward: true,
            is_prequantized: false,
        }
    }
}

/// Pre-loaded expert weights for MoE construction.
///
/// This struct holds already-loaded expert weights, allowing `MoEExperts`
/// to be constructed from weights loaded by any mechanism.
pub struct LoadedExpertWeights {
    /// Gate projection weights, one per expert.
    pub gate_proj: Vec<Arc<dyn QuantMethod>>,
    /// Up projection weights, one per expert.
    pub up_proj: Vec<Arc<dyn QuantMethod>>,
    /// Down projection weights, one per expert.
    pub down_proj: Vec<Arc<dyn QuantMethod>>,
    /// Properties of the loaded weights for backend selection.
    pub quant_properties: QuantProperties,
}

impl LoadedExpertWeights {
    /// Number of experts.
    pub fn num_experts(&self) -> usize {
        self.gate_proj.len()
    }

    /// Validate that all weight vectors have the same length.
    pub fn validate(&self) -> Result<()> {
        let n = self.gate_proj.len();
        if self.up_proj.len() != n || self.down_proj.len() != n {
            candle_core::bail!(
                "Expert weight count mismatch: gate={}, up={}, down={}",
                n,
                self.up_proj.len(),
                self.down_proj.len()
            );
        }
        Ok(())
    }
}

/// MoE-specific weight loading extensions.
///
/// This trait extends `WeightSource` with methods specific to loading
/// Mixture of Experts layers, including expert weight loading and
/// properties needed for backend selection.
pub trait MoEWeightSource: WeightSource {
    /// Get quantization properties for MoE backend selection.
    ///
    /// The returned properties inform which MoE execution backend
    /// (Fused, Fast, Slow) should be used.
    fn quant_properties(&self) -> QuantProperties;

    /// Load the router/gate weight for MoE.
    ///
    /// Default implementation delegates to `get_weight`.
    fn load_gate(&self, name: &str) -> Result<Arc<dyn QuantMethod>> {
        self.get_weight(name)
    }

    /// Load all expert weights for a standard MoE layer.
    ///
    /// Loads gate_proj, up_proj, down_proj for all experts.
    /// Override this if your format has optimized bulk loading
    /// (e.g., stacked expert weights).
    ///
    /// # Arguments
    ///
    /// * `num_experts` - Number of experts to load
    /// * `prefix` - Prefix for expert weights (e.g., "model.layers.0.mlp")
    fn load_experts(&self, num_experts: usize, prefix: &str) -> Result<LoadedExpertWeights> {
        let mut gate_proj = Vec::with_capacity(num_experts);
        let mut up_proj = Vec::with_capacity(num_experts);
        let mut down_proj = Vec::with_capacity(num_experts);

        for i in 0..num_experts {
            gate_proj.push(self.get_weight(&format!("{prefix}.experts.{i}.gate_proj.weight"))?);
            up_proj.push(self.get_weight(&format!("{prefix}.experts.{i}.up_proj.weight"))?);
            down_proj.push(self.get_weight(&format!("{prefix}.experts.{i}.down_proj.weight"))?);
        }

        Ok(LoadedExpertWeights {
            gate_proj,
            up_proj,
            down_proj,
            quant_properties: self.quant_properties(),
        })
    }

    /// Check if experts are in stacked format.
    ///
    /// Stacked format has tensors like `experts.gate_up_proj` with shape
    /// `[num_experts, hidden, intermediate*2]` rather than per-expert tensors.
    fn has_stacked_experts(&self, prefix: &str) -> bool {
        self.contains_tensor(&format!("{prefix}.experts.gate_up_proj"))
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quant_properties_unquantized() {
        let props = QuantProperties::unquantized();
        assert!(props.quant_format.is_none());
        assert!(props.uniform_quantization);
        assert!(props.supports_indexed_forward);
        assert!(!props.is_prequantized);
    }

    #[test]
    fn test_quant_properties_gguf_q4k() {
        let props = QuantProperties::gguf(candle_core::quantized::GgmlDType::Q4K);
        assert_eq!(
            props.quant_format,
            Some(candle_core::quantized::GgmlDType::Q4K)
        );
        assert!(props.supports_indexed_forward);
        assert!(props.is_prequantized);
    }

    #[test]
    fn test_quant_properties_gguf_unsupported() {
        // Q4_0 doesn't have indexed forward kernel
        let props = QuantProperties::gguf(candle_core::quantized::GgmlDType::Q4_0);
        assert!(!props.supports_indexed_forward);
    }

    #[test]
    fn test_loaded_expert_weights_validation() {
        let loaded = LoadedExpertWeights {
            gate_proj: vec![],
            up_proj: vec![],
            down_proj: vec![],
            quant_properties: QuantProperties::unquantized(),
        };
        assert!(loaded.validate().is_ok());
        assert_eq!(loaded.num_experts(), 0);
    }
}
