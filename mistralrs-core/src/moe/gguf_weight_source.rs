//! GGUF weight source implementation.
//!
//! This module provides `GgufWeightSource`, which implements the `WeightSource`
//! and `MoEWeightSource` traits for loading weights from GGUF files.

use std::io::{Read, Seek};
use std::sync::Arc;

use candle_core::quantized::gguf_file::Content;
use candle_core::quantized::GgmlDType;
use candle_core::{DType, Device, Result};
use mistralrs_quant::{GgufMatMul, QuantMethod, QuantMethodConfig};

use super::weight_source::{LoadedExpertWeights, MoEWeightSource, QuantProperties, WeightSource};

/// Weight source for GGUF quantized model files.
///
/// This struct provides uniform access to weights stored in GGUF format,
/// implementing both `WeightSource` and `MoEWeightSource` traits.
///
/// # Lifetime Parameters
///
/// - `'a` - Lifetime of the GGUF `Content` reference
/// - `'r` - Lifetime of the reader
///
/// # Type Parameters
///
/// - `R` - Reader type (must implement `Seek + Read`)
///
/// # Example
///
/// ```ignore
/// let content = Content::read(&mut reader)?;
/// let weights = GgufWeightSource::new(&content, &mut reader, &device, DType::BF16);
///
/// // Load a weight
/// let q_proj = weights.get_weight("model.layers.0.self_attn.q_proj.weight")?;
///
/// // Load MoE experts
/// let experts = weights.load_experts(8, "model.layers.0.mlp")?;
/// ```
pub struct GgufWeightSource<'a, 'r, R: Seek + Read + Send + Sync> {
    content: &'a Content,
    reader: &'r mut R,
    device: Device,
    dtype: DType,
    /// Cached quant dtype from first loaded tensor (for uniform quant detection)
    detected_quant_dtype: Option<GgmlDType>,
}

impl<'a, 'r, R: Seek + Read + Send + Sync> GgufWeightSource<'a, 'r, R> {
    /// Create a new GGUF weight source.
    ///
    /// # Arguments
    ///
    /// * `content` - Parsed GGUF file content (header + tensor info)
    /// * `reader` - Reader positioned at the GGUF file (for tensor data access)
    /// * `device` - Target device for loaded tensors
    /// * `dtype` - Default dtype for non-quantized operations
    pub fn new(content: &'a Content, reader: &'r mut R, device: &Device, dtype: DType) -> Self {
        Self {
            content,
            reader,
            device: device.clone(),
            dtype,
            detected_quant_dtype: None,
        }
    }

    /// Get the GGUF tensor info for introspection.
    pub fn tensor_info(&self, name: &str) -> Option<&candle_core::quantized::gguf_file::TensorInfo> {
        self.content.tensor_infos.get(name)
    }

    /// Get metadata from the GGUF file.
    pub fn metadata(&self) -> &std::collections::HashMap<String, candle_core::quantized::gguf_file::Value> {
        &self.content.metadata
    }

    /// Load a raw QTensor by name.
    fn load_qtensor(&mut self, name: &str) -> Result<candle_core::quantized::QTensor> {
        self.content.tensor(self.reader, name, &self.device)
    }
}

impl<R: Seek + Read + Send + Sync> WeightSource for GgufWeightSource<'_, '_, R> {
    fn get_weight(&self, name: &str) -> Result<Arc<dyn QuantMethod>> {
        // GGUF requires mutable reader for seeking, but WeightSource trait uses &self.
        // This is a design tension - for now we'll need interior mutability or
        // a different approach. Let's document this limitation.
        //
        // For actual use, we'll need to either:
        // 1. Use RefCell/Mutex for the reader
        // 2. Change the trait to use &mut self
        // 3. Pre-load all tensors
        //
        // For now, return an error indicating this limitation.
        candle_core::bail!(
            "GgufWeightSource::get_weight requires mutable access. \
             Use load_weight_mut() or pre-load tensors. Tensor: {name}"
        )
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.content.tensor_infos.contains_key(name)
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn dtype(&self) -> DType {
        self.dtype
    }

    fn get_raw_tensor(&self, name: &str) -> Result<candle_core::Tensor> {
        // Same limitation as get_weight - needs mutable reader
        candle_core::bail!(
            "GgufWeightSource::get_raw_tensor requires mutable access. Tensor: {name}"
        )
    }
}

impl<R: Seek + Read + Send + Sync> MoEWeightSource for GgufWeightSource<'_, '_, R> {
    fn quant_properties(&self) -> QuantProperties {
        // If we've detected a quant dtype, use it; otherwise assume Q4K as common default
        match self.detected_quant_dtype {
            Some(dtype) => QuantProperties::gguf(dtype),
            None => QuantProperties::gguf(GgmlDType::Q4K),
        }
    }

    fn load_experts(&self, _num_experts: usize, _prefix: &str) -> Result<LoadedExpertWeights> {
        // Same limitation - needs mutable access
        candle_core::bail!(
            "GgufWeightSource::load_experts requires mutable access. \
             Use load_experts_mut() instead."
        )
    }
}

// ============================================================================
// Mutable API (actual implementation)
// ============================================================================

impl<'a, 'r, R: Seek + Read + Send + Sync> GgufWeightSource<'a, 'r, R> {
    /// Load a weight tensor (mutable version).
    ///
    /// This is the actual implementation that requires mutable reader access.
    pub fn load_weight_mut(&mut self, name: &str) -> Result<Arc<dyn QuantMethod>> {
        let qtensor = self.load_qtensor(name)?;

        // Track detected quant dtype for uniform quant detection
        if self.detected_quant_dtype.is_none() {
            self.detected_quant_dtype = Some(qtensor.dtype());
        }

        Ok(Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
            q_weight: Arc::new(qtensor),
            b: None,
        })?) as Arc<dyn QuantMethod>)
    }

    /// Load a weight tensor with optional bias (mutable version).
    pub fn load_weight_with_bias_mut(
        &mut self,
        weight_name: &str,
        bias_name: Option<&str>,
    ) -> Result<Arc<dyn QuantMethod>> {
        let qtensor = self.load_qtensor(weight_name)?;

        // Track detected quant dtype
        if self.detected_quant_dtype.is_none() {
            self.detected_quant_dtype = Some(qtensor.dtype());
        }

        let bias = if let Some(bn) = bias_name {
            if self.contains_tensor(bn) {
                Some(self.load_qtensor(bn)?.dequantize(&self.device)?)
            } else {
                None
            }
        } else {
            None
        };

        Ok(Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
            q_weight: Arc::new(qtensor),
            b: bias,
        })?) as Arc<dyn QuantMethod>)
    }

    /// Load a raw tensor (dequantized) - mutable version.
    pub fn load_raw_tensor_mut(&mut self, name: &str) -> Result<candle_core::Tensor> {
        let qtensor = self.load_qtensor(name)?;
        qtensor.dequantize(&self.device)
    }

    /// Load all expert weights for a standard MoE layer (mutable version).
    ///
    /// # Arguments
    ///
    /// * `num_experts` - Number of experts to load
    /// * `prefix` - Prefix for expert weights (e.g., "blk.0")
    /// * `naming` - Tensor naming convention
    pub fn load_experts_mut(
        &mut self,
        num_experts: usize,
        prefix: &str,
        naming: &GgufExpertNaming,
    ) -> Result<LoadedExpertWeights> {
        let mut gate_proj = Vec::with_capacity(num_experts);
        let mut up_proj = Vec::with_capacity(num_experts);
        let mut down_proj = Vec::with_capacity(num_experts);

        // Check for stacked format first
        let stacked_gate = naming.stacked_gate(prefix);
        if self.contains_tensor(&stacked_gate) {
            return self.load_stacked_experts_mut(num_experts, prefix, naming);
        }

        // Load per-expert weights
        for i in 0..num_experts {
            let gate_name = naming.expert_gate(prefix, i);
            let up_name = naming.expert_up(prefix, i);
            let down_name = naming.expert_down(prefix, i);

            gate_proj.push(self.load_weight_mut(&gate_name)?);
            up_proj.push(self.load_weight_mut(&up_name)?);
            down_proj.push(self.load_weight_mut(&down_name)?);
        }

        Ok(LoadedExpertWeights {
            gate_proj,
            up_proj,
            down_proj,
            quant_properties: self.quant_properties_mut(),
        })
    }

    /// Load stacked expert weights and split them.
    fn load_stacked_experts_mut(
        &mut self,
        num_experts: usize,
        prefix: &str,
        naming: &GgufExpertNaming,
    ) -> Result<LoadedExpertWeights> {
        let gate_stacked = self.load_qtensor(&naming.stacked_gate(prefix))?;
        let up_stacked = self.load_qtensor(&naming.stacked_up(prefix))?;
        let down_stacked = self.load_qtensor(&naming.stacked_down(prefix))?;

        // Dequantize, chunk, re-quantize per expert
        // This is inefficient but necessary for loop-based execution
        let gate_dtype = gate_stacked.dtype();
        let up_dtype = up_stacked.dtype();
        let down_dtype = down_stacked.dtype();

        let gate_chunks = gate_stacked
            .dequantize(&self.device)?
            .chunk(num_experts, 0)?;
        let up_chunks = up_stacked
            .dequantize(&self.device)?
            .chunk(num_experts, 0)?;
        let down_chunks = down_stacked
            .dequantize(&self.device)?
            .chunk(num_experts, 0)?;

        let mut gate_proj = Vec::with_capacity(num_experts);
        let mut up_proj = Vec::with_capacity(num_experts);
        let mut down_proj = Vec::with_capacity(num_experts);

        for ((gate, up), down) in gate_chunks
            .into_iter()
            .zip(up_chunks)
            .zip(down_chunks)
        {
            use candle_core::quantized::QTensor;

            gate_proj.push(Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                q_weight: Arc::new(QTensor::quantize(&gate, gate_dtype)?),
                b: None,
            })?) as Arc<dyn QuantMethod>);

            up_proj.push(Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                q_weight: Arc::new(QTensor::quantize(&up, up_dtype)?),
                b: None,
            })?) as Arc<dyn QuantMethod>);

            down_proj.push(Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                q_weight: Arc::new(QTensor::quantize(&down, down_dtype)?),
                b: None,
            })?) as Arc<dyn QuantMethod>);
        }

        Ok(LoadedExpertWeights {
            gate_proj,
            up_proj,
            down_proj,
            quant_properties: self.quant_properties_mut(),
        })
    }

    /// Get quant properties (mutable version that updates detected dtype).
    fn quant_properties_mut(&self) -> QuantProperties {
        match self.detected_quant_dtype {
            Some(dtype) => QuantProperties::gguf(dtype),
            None => QuantProperties::gguf(GgmlDType::Q4K),
        }
    }

    /// Load the MoE router/gate weight.
    pub fn load_gate_mut(&mut self, name: &str) -> Result<Arc<dyn QuantMethod>> {
        self.load_weight_mut(name)
    }
}

// ============================================================================
// Tensor Naming Conventions
// ============================================================================

/// Naming convention for GGUF expert tensors.
///
/// Different GGUF models use different naming patterns for MoE layers.
/// This struct encapsulates the naming logic.
#[derive(Debug, Clone)]
pub struct GgufExpertNaming {
    /// Pattern for per-expert gate: "{prefix}.{pattern}.{expert_idx}.weight"
    pub gate_pattern: String,
    /// Pattern for per-expert up: "{prefix}.{pattern}.{expert_idx}.weight"
    pub up_pattern: String,
    /// Pattern for per-expert down: "{prefix}.{pattern}.{expert_idx}.weight"
    pub down_pattern: String,
    /// Pattern for stacked gate: "{prefix}.{pattern}.weight"
    pub stacked_gate_pattern: String,
    /// Pattern for stacked up: "{prefix}.{pattern}.weight"
    pub stacked_up_pattern: String,
    /// Pattern for stacked down: "{prefix}.{pattern}.weight"
    pub stacked_down_pattern: String,
}

impl Default for GgufExpertNaming {
    fn default() -> Self {
        Self::llama()
    }
}

impl GgufExpertNaming {
    /// Llama/Mixtral GGUF naming convention.
    ///
    /// Per-expert: `blk.{layer}.ffn_gate.{expert}.weight`
    /// Stacked: `blk.{layer}.ffn_gate_exps.weight`
    pub fn llama() -> Self {
        Self {
            gate_pattern: "ffn_gate".to_string(),
            up_pattern: "ffn_up".to_string(),
            down_pattern: "ffn_down".to_string(),
            stacked_gate_pattern: "ffn_gate_exps".to_string(),
            stacked_up_pattern: "ffn_up_exps".to_string(),
            stacked_down_pattern: "ffn_down_exps".to_string(),
        }
    }

    /// Qwen GGUF naming convention.
    pub fn qwen() -> Self {
        Self {
            gate_pattern: "ffn_gate".to_string(),
            up_pattern: "ffn_up".to_string(),
            down_pattern: "ffn_down".to_string(),
            stacked_gate_pattern: "ffn_gate_exps".to_string(),
            stacked_up_pattern: "ffn_up_exps".to_string(),
            stacked_down_pattern: "ffn_down_exps".to_string(),
        }
    }

    /// DeepSeek GGUF naming convention.
    pub fn deepseek() -> Self {
        Self {
            gate_pattern: "ffn_gate".to_string(),
            up_pattern: "ffn_up".to_string(),
            down_pattern: "ffn_down".to_string(),
            stacked_gate_pattern: "ffn_gate_exps".to_string(),
            stacked_up_pattern: "ffn_up_exps".to_string(),
            stacked_down_pattern: "ffn_down_exps".to_string(),
        }
    }

    /// Get per-expert gate tensor name.
    pub fn expert_gate(&self, prefix: &str, expert_idx: usize) -> String {
        format!("{prefix}.{}.{expert_idx}.weight", self.gate_pattern)
    }

    /// Get per-expert up tensor name.
    pub fn expert_up(&self, prefix: &str, expert_idx: usize) -> String {
        format!("{prefix}.{}.{expert_idx}.weight", self.up_pattern)
    }

    /// Get per-expert down tensor name.
    pub fn expert_down(&self, prefix: &str, expert_idx: usize) -> String {
        format!("{prefix}.{}.{expert_idx}.weight", self.down_pattern)
    }

    /// Get stacked gate tensor name.
    pub fn stacked_gate(&self, prefix: &str) -> String {
        format!("{prefix}.{}.weight", self.stacked_gate_pattern)
    }

    /// Get stacked up tensor name.
    pub fn stacked_up(&self, prefix: &str) -> String {
        format!("{prefix}.{}.weight", self.stacked_up_pattern)
    }

    /// Get stacked down tensor name.
    pub fn stacked_down(&self, prefix: &str) -> String {
        format!("{prefix}.{}.weight", self.stacked_down_pattern)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_llama_naming() {
        let naming = GgufExpertNaming::llama();
        assert_eq!(
            naming.expert_gate("blk.0", 3),
            "blk.0.ffn_gate.3.weight"
        );
        assert_eq!(
            naming.stacked_gate("blk.0"),
            "blk.0.ffn_gate_exps.weight"
        );
    }

    #[test]
    fn test_quant_properties_indexed_forward_support() {
        // Q4K should support indexed forward
        let props = QuantProperties::gguf(GgmlDType::Q4K);
        assert!(props.supports_indexed_forward);

        // Q4_0 should not (no optimized kernel)
        let props = QuantProperties::gguf(GgmlDType::Q4_0);
        assert!(!props.supports_indexed_forward);
    }
}
