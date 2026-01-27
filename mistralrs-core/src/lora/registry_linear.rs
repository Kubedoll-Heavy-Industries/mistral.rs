//! Registry-based LoRA linear layer.
//!
//! This module provides [`RegistryLoraLinear`], a linear layer that fetches
//! adapter weights from a shared [`AdapterRegistry`] rather than storing them
//! internally. This enables runtime adapter switching without rebuilding the layer.

use std::sync::{Arc, RwLock};

use candle_core::{DType, Result, Tensor};
use mistralrs_quant::QuantMethod;

use super::{AdapterRegistry, LinearLayerLike, Merge};
use crate::layers::MatMul;

/// Cache for stacked adapter tensors.
///
/// When multiple adapters are active with compatible shapes, we can stack their
/// weights into a single tensor for batched matrix multiplication. This cache
/// stores the stacked tensors and invalidates when the active adapter set changes.
#[derive(Debug)]
struct StackedAdapterCache {
    /// Active adapter names when this cache was built.
    active_set: Vec<String>,
    /// Stacked A matrices: (num_adapters, rank, in_features)
    a_stacked: Tensor,
    /// Stacked B matrices: (num_adapters, out_features, rank)
    b_stacked: Tensor,
    /// Scale factors for each adapter.
    scales: Vec<f64>,
}

/// A LoRA linear layer that fetches weights from a shared adapter registry.
///
/// Unlike [`LoraLinear`](super::LoraLinear) which stores adapter weights internally,
/// this layer references a shared [`AdapterRegistry`] and fetches the currently
/// active adapter weights during forward pass. This enables:
///
/// - Runtime adapter switching without layer reconstruction
/// - Memory efficiency (adapters shared across layers)
/// - Per-request adapter selection
///
/// # Caching
///
/// To avoid repeated weight fetching, the layer caches stacked adapter tensors.
/// The cache is automatically invalidated when the active adapter set changes.
///
/// # Example
///
/// ```ignore
/// let registry = Arc::new(AdapterRegistry::new(device));
/// registry.register("style", config, weights)?;
///
/// let layer = RegistryLoraLinear::new(base_linear, registry.clone(), 0);
///
/// // Activate adapter
/// registry.set_active(&["style"])?;
///
/// // Forward uses "style" adapter
/// let output = layer.lora_forward(&input, None, 1.0, None)?;
///
/// // Switch adapter without rebuilding layer
/// registry.set_active(&["other"])?;
/// let output2 = layer.lora_forward(&input, None, 1.0, None)?;
/// ```
#[derive(Debug)]
pub struct RegistryLoraLinear {
    /// Base linear layer (may be quantized).
    base: Arc<dyn QuantMethod>,
    /// Shared adapter registry.
    registry: Arc<AdapterRegistry>,
    /// Which layer this is (for fetching correct weights).
    layer_idx: usize,
    /// Cached stacked tensors for current active set.
    cache: RwLock<Option<StackedAdapterCache>>,
    /// Whether adapters have been merged into base weights.
    merged: bool,
}

impl RegistryLoraLinear {
    /// Create a new registry-based LoRA linear layer.
    ///
    /// # Arguments
    ///
    /// * `base` - The base linear layer (can be quantized)
    /// * `registry` - Shared adapter registry
    /// * `layer_idx` - Index of this layer (for fetching weights)
    pub fn new(base: Arc<dyn QuantMethod>, registry: Arc<AdapterRegistry>, layer_idx: usize) -> Self {
        Self {
            base,
            registry,
            layer_idx,
            cache: RwLock::new(None),
            merged: false,
        }
    }

    /// Check if the cache is valid for the current active adapter set.
    fn is_cache_valid(&self, current_active: &[String]) -> bool {
        if let Ok(cache) = self.cache.read() {
            if let Some(ref cached) = *cache {
                return cached.active_set == current_active;
            }
        }
        false
    }

    /// Build stacked tensors from current active adapters.
    fn build_cache(&self) -> Result<Option<StackedAdapterCache>> {
        let active_names = self.registry.get_active_names()?;
        if active_names.is_empty() {
            return Ok(None);
        }

        let weights = self.registry.get_active_weights_for_layer(self.layer_idx)?;
        if weights.is_empty() {
            return Ok(None);
        }

        // Check if all adapters have compatible shapes for stacking
        let first_a_shape = weights[0].0.dims();
        let first_b_shape = weights[0].1.dims();
        let can_stack = weights.iter().all(|(a, b, _)| {
            a.dims() == first_a_shape && b.dims() == first_b_shape
        });

        if !can_stack || weights.len() == 1 {
            // Can't stack or only one adapter - no benefit from stacking
            // We'll handle this in forward by iterating
            return Ok(None);
        }

        // Stack A matrices: each is (rank, in_features) -> (num_adapters, rank, in_features)
        let a_tensors: Vec<Tensor> = weights
            .iter()
            .map(|(a, _, _)| a.unsqueeze(0))
            .collect::<Result<Vec<_>>>()?;
        let a_stacked = Tensor::cat(&a_tensors, 0)?;

        // Stack B matrices: each is (out_features, rank) -> (num_adapters, out_features, rank)
        let b_tensors: Vec<Tensor> = weights
            .iter()
            .map(|(_, b, _)| b.unsqueeze(0))
            .collect::<Result<Vec<_>>>()?;
        let b_stacked = Tensor::cat(&b_tensors, 0)?;

        let scales: Vec<f64> = weights.iter().map(|(_, _, s)| *s).collect();

        // Pre-multiply A by scales for efficiency
        #[allow(clippy::cast_possible_truncation)]
        let scale_tensor = Tensor::from_vec(
            scales.iter().map(|&s| s as f32).collect::<Vec<_>>(),
            (scales.len(), 1, 1),
            a_stacked.device(),
        )?
        .to_dtype(a_stacked.dtype())?;
        let a_stacked = a_stacked.broadcast_mul(&scale_tensor)?;

        Ok(Some(StackedAdapterCache {
            active_set: active_names,
            a_stacked,
            b_stacked,
            scales,
        }))
    }

    /// Forward pass using stacked tensors (fast path).
    fn forward_stacked(
        &self,
        input: &Tensor,
        cache: &StackedAdapterCache,
        global_scaling_weight: f64,
    ) -> Result<Tensor> {
        let (b, s, h) = input.dims3()?;
        let input_flat = input.reshape((b * s, h))?;

        // Batched matmul: (n_adapters, rank, in) @ (in, b*s) -> (n_adapters, rank, b*s)
        let out = cache.a_stacked.broadcast_matmul(&input_flat.t()?)?;
        // (n_adapters, out, rank) @ (n_adapters, rank, b*s) -> (n_adapters, out, b*s)
        let out = cache.b_stacked.broadcast_matmul(&out)?;

        let o_h = out.dims()[1];
        let out = out.reshape((cache.scales.len(), b, s, o_h))?;

        // Sum over adapters and apply global scaling
        Ok((out.sum(0)? * global_scaling_weight)?)
    }

    /// Forward pass iterating over adapters (slow path for incompatible shapes).
    fn forward_iterate(
        &self,
        input: &Tensor,
        weights: &[(Tensor, Tensor, f64)],
        global_scaling_weight: f64,
    ) -> Result<Tensor> {
        let (b_size, s, h) = input.dims3()?;
        let input_flat = input.reshape((b_size * s, h))?;

        let mut result: Option<Tensor> = None;

        for (a, b_mat, scale) in weights {
            // input_flat: (batch*seq, hidden)
            // a: (rank, hidden) -> input @ a.T gives (batch*seq, rank)
            // b: (out, rank) -> (batch*seq, rank) @ b.T gives (batch*seq, out)
            let input_dtype = input_flat.to_dtype(a.dtype())?;
            let adapter_out = input_dtype.matmul(&a.t()?)?.matmul(&b_mat.t()?)?;
            let scaled = ((adapter_out * *scale)? * global_scaling_weight)?;

            result = Some(match result {
                Some(r) => (r + scaled)?,
                None => scaled,
            });
        }

        // Reshape back to (batch, seq, out_features)
        let result = result.ok_or_else(|| {
            candle_core::Error::Msg("No adapter weights to iterate".to_string())
        })?;
        let out_features = result.dim(1)?;
        result.reshape((b_size, s, out_features))
    }
}

impl Merge for RegistryLoraLinear {
    fn get_delta_weight(&self, adapter_idx: usize) -> Result<Tensor> {
        let weights = self.registry.get_active_weights_for_layer(self.layer_idx)?;
        if adapter_idx >= weights.len() {
            return Err(candle_core::Error::Msg(format!(
                "Adapter index {} out of range (have {} active)",
                adapter_idx,
                weights.len()
            )));
        }

        let (a, b, scale) = &weights[adapter_idx];
        Ok((MatMul.matmul(b, a)? * *scale)?)
    }

    fn merge_weights(&mut self) -> Result<()> {
        let weights = self.registry.get_active_weights_for_layer(self.layer_idx)?;
        if weights.is_empty() {
            return Ok(());
        }

        let mut delta: Option<Tensor> = None;
        for (idx, _) in weights.iter().enumerate() {
            let d = self.get_delta_weight(idx)?;
            delta = Some(match delta {
                Some(existing) => (existing + d)?,
                None => d,
            });
        }

        if let Some(d) = delta {
            // add_delta_w returns a new Arc with merged weights
            self.base = self.base.add_delta_w(&d)?;
        }

        self.merged = true;
        Ok(())
    }
}

impl LinearLayerLike for RegistryLoraLinear {
    fn quant_inner(&mut self) -> &mut Arc<dyn QuantMethod> {
        &mut self.base
    }

    fn bias(&self) -> Option<&Tensor> {
        // Registry-based layers don't expose bias directly
        None
    }

    fn weight(&self) -> &Tensor {
        unreachable!("Registry-based layers don't expose weight directly")
    }

    fn quantized_act_type(&self) -> Option<DType> {
        self.base.quantized_act_type()
    }

    fn lora_forward(
        &self,
        input: &Tensor,
        _scalings: Option<Tensor>, // XLoRA scalings - handled separately
        global_scaling_weight: f64,
        is_scaling_pass: Option<f64>,
    ) -> Result<Tensor> {
        // Base forward
        let result = self.base.forward(input)?;

        // If merged, adapters are already in base weights
        if self.merged {
            return Ok(result);
        }

        // Scaling pass with weight 0 means skip adapter application
        if is_scaling_pass.is_some_and(|x| x == 0.) {
            return Ok(result);
        }

        // Get current active adapters
        let active_names = self.registry.get_active_names()?;
        if active_names.is_empty() {
            return Ok(result);
        }

        // Check cache validity
        if self.is_cache_valid(&active_names) {
            if let Ok(cache_guard) = self.cache.read() {
                if let Some(ref cache) = *cache_guard {
                    let adapter_out = self.forward_stacked(input, cache, global_scaling_weight)?;
                    return result + adapter_out;
                }
            }
        }

        // Rebuild cache
        let new_cache = self.build_cache()?;

        if let Some(ref cache) = new_cache {
            // Use stacked fast path
            let adapter_out = self.forward_stacked(input, cache, global_scaling_weight)?;

            // Store cache for next time
            if let Ok(mut cache_guard) = self.cache.write() {
                *cache_guard = new_cache;
            }

            result + adapter_out
        } else {
            // Use iterate slow path
            let weights = self.registry.get_active_weights_for_layer(self.layer_idx)?;
            if weights.is_empty() {
                return Ok(result);
            }

            let adapter_out = self.forward_iterate(input, &weights, global_scaling_weight)?;
            result + adapter_out
        }
    }

    fn is_lora(&self) -> bool {
        // Check if registry has any adapters for this layer
        self.registry
            .get_active_weights_for_layer(self.layer_idx)
            .map(|w| !w.is_empty())
            .unwrap_or(false)
    }
}

// ============================================================================
// QuantMethod Implementation
// ============================================================================
//
// This allows RegistryLoraLinear to be used as a drop-in replacement for any
// Arc<dyn QuantMethod>, enabling transparent per-request adapter switching.

impl mistralrs_quant::QuantizedSerde for RegistryLoraLinear {
    fn name(&self) -> &'static str {
        "registry_lora_linear"
    }

    fn isq_serde_supported(&self) -> bool {
        // Serialization not supported - registry state is runtime-only
        false
    }
}

impl mistralrs_quant::QuantMethod for RegistryLoraLinear {
    fn new(_method: mistralrs_quant::QuantMethodConfig) -> Result<Self>
    where
        Self: Sized,
    {
        // RegistryLoraLinear is constructed via RegistryLoraLinear::new(), not this method
        candle_core::bail!(
            "RegistryLoraLinear cannot be constructed via QuantMethod::new(). \
             Use RegistryLoraLinear::new(base, registry, layer_idx) instead."
        )
    }

    fn dequantize_w(&self) -> Result<Tensor> {
        // Delegate to base layer - LoRA weights are separate
        self.base.dequantize_w()
    }

    fn forward(&self, a: &Tensor) -> Result<Tensor> {
        // Use lora_forward with default scaling parameters
        // global_scaling_weight=1.0 means full adapter effect
        // is_scaling_pass=None means normal forward (not XLoRA scaling)
        self.lora_forward(a, None, 1.0, None)
    }

    fn quantized_act_type(&self) -> Option<DType> {
        // Delegate to base layer
        self.base.quantized_act_type()
    }

    fn dtype_and_device(&self) -> (DType, candle_core::Device) {
        // Delegate to base layer
        self.base.dtype_and_device()
    }

    fn add_delta_w(&self, delta: &Tensor) -> Result<Arc<dyn QuantMethod>> {
        // Add delta to base weights - this is for static LoRA merging
        // For dynamic LoRA, we don't merge but apply at forward time
        let new_base = self.base.add_delta_w(delta)?;
        Ok(Arc::new(RegistryLoraLinear {
            base: new_base,
            registry: self.registry.clone(),
            layer_idx: self.layer_idx,
            cache: RwLock::new(None),
            merged: self.merged,
        }))
    }

    fn apply_isq(
        self: Arc<Self>,
        dtype: Option<mistralrs_quant::IsqType>,
        device: candle_core::Device,
        n_quantized: &std::sync::atomic::AtomicUsize,
        imatrix_weight: Option<Vec<f32>>,
        guard: mistralrs_quant::QuantizeOntoGuard,
    ) -> Result<Arc<dyn QuantMethod>> {
        // Apply ISQ to base layer, rewrap with LoRA
        let new_base = self
            .base
            .clone()
            .apply_isq(dtype, device, n_quantized, imatrix_weight, guard)?;
        Ok(Arc::new(RegistryLoraLinear {
            base: new_base,
            registry: self.registry.clone(),
            layer_idx: self.layer_idx,
            cache: RwLock::new(None),
            merged: self.merged,
        }))
    }
}

/// Wrap a base linear layer with LoRA adapter support.
///
/// This creates a `RegistryLoraLinear` that can be used anywhere an
/// `Arc<dyn QuantMethod>` is expected. The returned layer will apply
/// active adapters from the registry during forward pass.
///
/// # Arguments
///
/// * `base` - Base linear layer to wrap
/// * `registry` - Adapter registry for fetching active adapters
/// * `layer_idx` - Index of this layer in the model (for fetching correct weights)
///
/// # Example
///
/// ```ignore
/// let registry = Arc::new(AdapterRegistry::new(device));
/// let base_layer = load_linear_layer(...)?;
/// let lora_layer = wrap_with_lora(base_layer, registry.clone(), 0);
/// ```
pub fn wrap_with_lora(
    base: Arc<dyn QuantMethod>,
    registry: Arc<AdapterRegistry>,
    layer_idx: usize,
) -> Arc<dyn QuantMethod> {
    Arc::new(RegistryLoraLinear::new(base, registry, layer_idx))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lora::AdapterWeights;
    use candle_core::Device;
    // Use LoraConfig from mistralrs_quant (re-exported via registry module)
    use mistralrs_quant::{LoraConfig, QuantMethodConfig, UnquantLinear};
    use std::collections::HashSet;

    fn test_config() -> LoraConfig {
        LoraConfig {
            rank: 4,
            alpha: 8.0,
            target_modules: HashSet::from(["test".to_string()]),
        }
    }

    fn create_test_layer() -> Result<(RegistryLoraLinear, Arc<AdapterRegistry>)> {
        let device = Device::Cpu;
        let in_features = 16;
        let out_features = 32;

        // Create base linear
        let weight = Tensor::randn(0.0f32, 1.0, (out_features, in_features), &device)?;
        let base = Arc::new(UnquantLinear::new(QuantMethodConfig::Unquantized(
            candle_nn::Linear::new(weight, None),
        ))?) as Arc<dyn QuantMethod>;

        // Create registry
        let registry = Arc::new(AdapterRegistry::new(device.clone()));

        // Create layer
        let layer = RegistryLoraLinear::new(base, registry.clone(), 0);

        Ok((layer, registry))
    }

    #[test]
    fn test_forward_no_adapters() -> Result<()> {
        let (layer, _registry) = create_test_layer()?;

        let input = Tensor::randn(0.0f32, 1.0, (1, 8, 16), &Device::Cpu)?;
        let output = layer.lora_forward(&input, None, 1.0, None)?;

        // Should just be base forward
        assert_eq!(output.dims(), &[1, 8, 32]);
        Ok(())
    }

    #[test]
    fn test_forward_with_adapter() -> Result<()> {
        let (layer, registry) = create_test_layer()?;
        let device = Device::Cpu;

        // Create adapter weights for layer 0
        let mut weights = AdapterWeights::new();
        let a = Tensor::randn(0.0f32, 0.1, (4, 16), &device)?; // rank=4, in=16
        let b = Tensor::randn(0.0f32, 0.1, (32, 4), &device)?; // out=32, rank=4
        weights.add_layer(0, a, b);

        registry.register("test-adapter", test_config(), weights)?;
        registry.set_active(&["test-adapter"])?;

        let input = Tensor::randn(0.0f32, 1.0, (1, 8, 16), &device)?;
        let output = layer.lora_forward(&input, None, 1.0, None)?;

        assert_eq!(output.dims(), &[1, 8, 32]);
        Ok(())
    }

    #[test]
    fn test_cache_invalidation() -> Result<()> {
        let (layer, registry) = create_test_layer()?;
        let device = Device::Cpu;

        // Register three adapters (need 2+ active to trigger stacking cache)
        let mut weights1 = AdapterWeights::new();
        weights1.add_layer(
            0,
            Tensor::randn(0.0f32, 0.1, (4, 16), &device)?,
            Tensor::randn(0.0f32, 0.1, (32, 4), &device)?,
        );
        registry.register("adapter-1", test_config(), weights1)?;

        let mut weights2 = AdapterWeights::new();
        weights2.add_layer(
            0,
            Tensor::randn(0.0f32, 0.1, (4, 16), &device)?,
            Tensor::randn(0.0f32, 0.1, (32, 4), &device)?,
        );
        registry.register("adapter-2", test_config(), weights2)?;

        let mut weights3 = AdapterWeights::new();
        weights3.add_layer(
            0,
            Tensor::randn(0.0f32, 0.1, (4, 16), &device)?,
            Tensor::randn(0.0f32, 0.1, (32, 4), &device)?,
        );
        registry.register("adapter-3", test_config(), weights3)?;

        let input = Tensor::randn(0.0f32, 1.0, (1, 8, 16), &device)?;

        // Activate both adapters (triggers stacking cache)
        registry.set_active(&["adapter-1", "adapter-2"])?;
        let _out1 = layer.lora_forward(&input, None, 1.0, None)?;

        // Cache should be valid for current active set
        assert!(layer.is_cache_valid(&["adapter-1".to_string(), "adapter-2".to_string()]));
        assert!(!layer.is_cache_valid(&["adapter-2".to_string(), "adapter-3".to_string()]));

        // Switch to different adapter set
        registry.set_active(&["adapter-2", "adapter-3"])?;
        let _out2 = layer.lora_forward(&input, None, 1.0, None)?;

        // Cache should now be for new set
        assert!(!layer.is_cache_valid(&["adapter-1".to_string(), "adapter-2".to_string()]));
        assert!(layer.is_cache_valid(&["adapter-2".to_string(), "adapter-3".to_string()]));

        Ok(())
    }
}
