//! Thread-safe, model-scoped adapter registry for runtime adapter management.
//!
//! This module provides [`AdapterRegistry`], which replaces the thread-local
//! LoRA registry with a thread-safe, model-scoped solution that enables:
//!
//! - Runtime adapter switching without model reload
//! - Per-request adapter selection
//! - XLoRA per-token scalings (via [`AdapterSelection::Scalings`])
//! - Lazy loading and offloading of adapters
//! - Multiple concurrent requests with different adapter sets
//!
//! # Example
//!
//! ```ignore
//! let registry = AdapterRegistry::new(device);
//!
//! // Load adapters
//! registry.register("style-formal", config, weights)?;
//! registry.register("skill-coding", config, weights)?;
//!
//! // Standard LoRA: activate specific adapters
//! registry.set_selection(AdapterSelection::Active(vec!["style-formal".into()]));
//!
//! // XLoRA: use per-token scalings from classifier
//! let scalings = classifier.forward(&hidden_states)?;
//! registry.set_selection(AdapterSelection::Scalings(scalings));
//! ```

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::RwLock;

use candle_core::{Device, Result, Tensor};

// ============================================================================
// Adapter Selection Strategy
// ============================================================================

/// Strategy for selecting/weighting adapters during forward pass.
///
/// This is runtime state (changes per-request), not compile-time configuration.
/// XLoRA is not a separate mode—it's just a different selection strategy.
///
/// # Variants
///
/// - [`Active`](Self::Active): Standard LoRA with named adapters at equal weight
/// - [`Scalings`](Self::Scalings): XLoRA with per-token scalings from classifier
#[derive(Debug, Clone)]
pub enum AdapterSelection {
    /// Per-request selection: named adapters with equal weight.
    ///
    /// Used for standard LoRA and per-request adapter switching.
    /// Adapters are applied in order, each with weight 1.0 * scale.
    Active(Vec<String>),

    /// Per-token scalings from XLoRA classifier.
    ///
    /// Shape: `[batch, seq, n_adapters]` for global scalings, or
    ///        `[batch, seq, n_layers, n_adapters]` for layerwise scalings.
    ///
    /// The adapter order matches [`AdapterRegistry::adapter_order`].
    Scalings(Tensor),
}

impl Default for AdapterSelection {
    fn default() -> Self {
        Self::Active(Vec::new())
    }
}

// Use the LoraConfig from mistralrs-quant which has public fields
pub use mistralrs_quant::LoraConfig;

/// Load state for an adapter's weights.
#[derive(Debug, Clone)]
pub enum AdapterLoadState {
    /// Weights loaded on GPU, ready for inference.
    Ready,
    /// Weights offloaded to CPU to save GPU memory.
    Offloaded,
    /// Adapter registered but weights not yet loaded.
    /// Contains the path to load from when needed.
    Deferred { path: PathBuf },
}

/// Weights for a single adapter, organized by layer.
#[derive(Debug, Clone)]
pub struct AdapterWeights {
    /// A matrices (down projection) by layer index.
    /// Shape: (rank, in_features)
    pub a_weights: HashMap<usize, Tensor>,
    /// B matrices (up projection) by layer index.
    /// Shape: (out_features, rank)
    pub b_weights: HashMap<usize, Tensor>,
}

impl AdapterWeights {
    /// Create new empty adapter weights.
    pub fn new() -> Self {
        Self {
            a_weights: HashMap::new(),
            b_weights: HashMap::new(),
        }
    }

    /// Add weights for a specific layer.
    pub fn add_layer(&mut self, layer_idx: usize, a: Tensor, b: Tensor) {
        self.a_weights.insert(layer_idx, a);
        self.b_weights.insert(layer_idx, b);
    }

    /// Get weights for a specific layer.
    pub fn get_layer(&self, layer_idx: usize) -> Option<(&Tensor, &Tensor)> {
        match (
            self.a_weights.get(&layer_idx),
            self.b_weights.get(&layer_idx),
        ) {
            (Some(a), Some(b)) => Some((a, b)),
            _ => None,
        }
    }

    /// Move all weights to CPU.
    pub fn move_to_cpu(&mut self) -> Result<()> {
        let cpu = Device::Cpu;
        for a in self.a_weights.values_mut() {
            *a = a.to_device(&cpu)?;
        }
        for b in self.b_weights.values_mut() {
            *b = b.to_device(&cpu)?;
        }
        Ok(())
    }

    /// Move all weights to the specified device.
    pub fn move_to_device(&mut self, device: &Device) -> Result<()> {
        for a in self.a_weights.values_mut() {
            *a = a.to_device(device)?;
        }
        for b in self.b_weights.values_mut() {
            *b = b.to_device(device)?;
        }
        Ok(())
    }
}

impl Default for AdapterWeights {
    fn default() -> Self {
        Self::new()
    }
}

/// Projection name to offset mapping for the unified layer index scheme.
///
/// Each transformer layer has 7 projections:
/// - q_proj (0), k_proj (1), v_proj (2), o_proj (3) for attention
/// - gate_proj (4), up_proj (5), down_proj (6) for MLP
pub const PROJECTION_OFFSETS: &[(&str, usize)] = &[
    ("q_proj", 0),
    ("k_proj", 1),
    ("v_proj", 2),
    ("o_proj", 3),
    ("gate_proj", 4),
    ("up_proj", 5),
    ("down_proj", 6),
];

/// Number of LoRA-targetable projections per transformer layer.
/// Used to compute unified indices: `layer_idx * PROJECTIONS_PER_LAYER + projection_offset`
pub const PROJECTIONS_PER_LAYER: usize = 7;

impl AdapterWeights {
    /// Parse LoRA weights from a VarBuilder into the unified index scheme.
    ///
    /// This function extracts A and B weight tensors from the VarBuilder and
    /// maps them to the unified layer index scheme used by `RegistryLoraLinear`:
    /// `unified_idx = layer_idx * 7 + projection_offset`
    ///
    /// # Arguments
    ///
    /// * `vb` - VarBuilder containing the adapter weights (from safetensors)
    /// * `num_layers` - Number of transformer layers in the model
    /// * `target_modules` - Which projections this adapter targets (e.g., ["q_proj", "v_proj"])
    ///
    /// # Weight Name Patterns
    ///
    /// Tries multiple common LoRA weight naming conventions:
    /// - `base_model.model.layers.{layer}.self_attn.{proj}.lora_{A|B}.weight`
    /// - `model.layers.{layer}.self_attn.{proj}.lora_{A|B}.weight`
    /// - `layers.{layer}.self_attn.{proj}.lora_{A|B}.weight` (fallback for GGUF-converted)
    ///
    /// MLP projections use similar patterns with `mlp.{proj}` instead of `self_attn.{proj}`.
    pub fn from_varbuilder(
        vb: &mistralrs_quant::ShardedVarBuilder,
        num_layers: usize,
        target_modules: &std::collections::HashSet<String>,
    ) -> Result<Self> {
        let mut weights = Self::new();

        // Common prefixes used by different LoRA libraries/formats
        let prefixes = ["base_model.model.", "model.", ""];

        // Attention vs MLP projection mapping
        let proj_to_submodule = |proj: &str| -> &'static str {
            match proj {
                "q_proj" | "k_proj" | "v_proj" | "o_proj" => "self_attn",
                "gate_proj" | "up_proj" | "down_proj" => "mlp",
                _ => "self_attn", // fallback
            }
        };

        for layer_idx in 0..num_layers {
            for (proj_name, proj_offset) in PROJECTION_OFFSETS {
                // Skip projections not targeted by this adapter
                if !target_modules.contains(*proj_name) {
                    continue;
                }

                let submodule = proj_to_submodule(proj_name);
                let unified_idx = layer_idx * 7 + proj_offset;

                // Try different prefix patterns
                let mut a_tensor: Option<Tensor> = None;
                let mut b_tensor: Option<Tensor> = None;

                for prefix in &prefixes {
                    let a_key =
                        format!("{prefix}layers.{layer_idx}.{submodule}.{proj_name}.lora_A.weight");
                    let b_key =
                        format!("{prefix}layers.{layer_idx}.{submodule}.{proj_name}.lora_B.weight");

                    // Try to load A tensor
                    if a_tensor.is_none() && vb.contains_tensor(&a_key) {
                        if let Ok(t) = vb.get_with_hints_dtype(
                            (),
                            &a_key,
                            Default::default(), // Full tensor (world_size=1)
                            candle_core::DType::F32,
                        ) {
                            a_tensor = Some(t);
                        }
                    }

                    // Try to load B tensor
                    if b_tensor.is_none() && vb.contains_tensor(&b_key) {
                        if let Ok(t) = vb.get_with_hints_dtype(
                            (),
                            &b_key,
                            Default::default(), // Full tensor (world_size=1)
                            candle_core::DType::F32,
                        ) {
                            b_tensor = Some(t);
                        }
                    }

                    // Break early if we found both
                    if a_tensor.is_some() && b_tensor.is_some() {
                        break;
                    }
                }

                // Add layer weights if we found both A and B
                if let (Some(a), Some(b)) = (a_tensor, b_tensor) {
                    weights.add_layer(unified_idx, a, b);
                }
            }
        }

        Ok(weights)
    }
}

/// A loaded adapter with its configuration and weights.
#[derive(Debug)]
pub struct LoadedAdapter {
    /// Adapter configuration (rank, alpha, target modules).
    pub config: LoraConfig,
    /// Adapter weights organized by layer.
    pub weights: AdapterWeights,
    /// Current load state.
    pub load_state: AdapterLoadState,
    /// Pre-computed scale factor (alpha / rank).
    pub scale: f64,
}

impl LoadedAdapter {
    /// Create a new loaded adapter.
    pub fn new(config: LoraConfig, weights: AdapterWeights) -> Self {
        let scale = if config.rank > 0 {
            config.alpha / config.rank as f64
        } else {
            1.0
        };
        Self {
            config,
            weights,
            load_state: AdapterLoadState::Ready,
            scale,
        }
    }

    /// Create a deferred adapter that will be loaded on first use.
    pub fn deferred(config: LoraConfig, path: PathBuf) -> Self {
        let scale = if config.rank > 0 {
            config.alpha / config.rank as f64
        } else {
            1.0
        };
        Self {
            config,
            weights: AdapterWeights::new(),
            load_state: AdapterLoadState::Deferred { path },
            scale,
        }
    }
}

/// Thread-safe adapter registry scoped to a model instance.
///
/// The registry manages the lifecycle of LoRA adapters:
/// - Registration (loading adapter weights)
/// - Selection (choosing which adapters to use and how to weight them)
/// - Offloading (moving weights to CPU to save GPU memory)
/// - Removal (unloading adapters entirely)
///
/// # Selection Strategies
///
/// The registry supports two selection strategies via [`AdapterSelection`]:
/// - **Active**: Standard LoRA with named adapters at equal weight
/// - **Scalings**: XLoRA with per-token scalings from a classifier
///
/// # Thread Safety
///
/// All operations are protected by `RwLock`, allowing concurrent reads
/// (e.g., multiple inference threads reading active adapters) with
/// exclusive writes (e.g., switching selection strategy).
#[derive(Debug)]
pub struct AdapterRegistry {
    /// All registered adapters by name.
    adapters: RwLock<HashMap<String, LoadedAdapter>>,
    /// Consistent ordering of adapter names for XLoRA scalings.
    /// When using `AdapterSelection::Scalings`, scalings[i] corresponds to adapter_order[i].
    adapter_order: RwLock<Vec<String>>,
    /// Current selection strategy.
    selection: RwLock<AdapterSelection>,
    /// Default selection to restore after per-request overrides.
    default_selection: RwLock<AdapterSelection>,
    /// Device for loading/moving weights.
    device: Device,
}

impl AdapterRegistry {
    /// Create a new adapter registry for the given device.
    pub fn new(device: Device) -> Self {
        Self {
            adapters: RwLock::new(HashMap::new()),
            adapter_order: RwLock::new(Vec::new()),
            selection: RwLock::new(AdapterSelection::default()),
            default_selection: RwLock::new(AdapterSelection::default()),
            device,
        }
    }

    /// Register an adapter with pre-loaded weights.
    ///
    /// The adapter is added to the registry but not selected.
    /// Call [`set_selection`](Self::set_selection) or [`set_active`](Self::set_active) to use it.
    ///
    /// The adapter is also added to [`adapter_order`](Self::adapter_order) for XLoRA scalings.
    pub fn register(
        &self,
        name: impl Into<String>,
        config: LoraConfig,
        weights: AdapterWeights,
    ) -> Result<()> {
        let name = name.into();
        let adapter = LoadedAdapter::new(config, weights);

        // Add to adapters map
        let mut adapters = self
            .adapters
            .write()
            .map_err(|e| candle_core::Error::Msg(format!("Failed to acquire write lock: {e}")))?;
        let is_new = !adapters.contains_key(&name);
        adapters.insert(name.clone(), adapter);
        drop(adapters);

        // Add to adapter order if new (for XLoRA scalings consistency)
        if is_new {
            let mut order = self
                .adapter_order
                .write()
                .map_err(|e| candle_core::Error::Msg(format!("Failed to acquire write lock: {e}")))?;
            order.push(name);
        }

        Ok(())
    }

    /// Register an adapter for deferred loading.
    ///
    /// Weights will be loaded from the given path on first use.
    pub fn register_deferred(
        &self,
        name: impl Into<String>,
        config: LoraConfig,
        path: PathBuf,
    ) -> Result<()> {
        let name = name.into();
        let adapter = LoadedAdapter::deferred(config, path);

        // Add to adapters map
        let mut adapters = self
            .adapters
            .write()
            .map_err(|e| candle_core::Error::Msg(format!("Failed to acquire write lock: {e}")))?;
        let is_new = !adapters.contains_key(&name);
        adapters.insert(name.clone(), adapter);
        drop(adapters);

        // Add to adapter order if new
        if is_new {
            let mut order = self
                .adapter_order
                .write()
                .map_err(|e| candle_core::Error::Msg(format!("Failed to acquire write lock: {e}")))?;
            order.push(name);
        }

        Ok(())
    }

    // ========================================================================
    // Selection Strategy
    // ========================================================================

    /// Set the adapter selection strategy.
    ///
    /// This is the primary method for controlling which adapters are used:
    /// - `Active(names)`: Use specific named adapters with equal weight (standard LoRA)
    /// - `Scalings(tensor)`: Use per-token scalings from XLoRA classifier
    ///
    /// For convenience, use [`set_active`](Self::set_active) for standard LoRA.
    pub fn set_selection(&self, selection: AdapterSelection) -> Result<()> {
        // Validate Active selection
        if let AdapterSelection::Active(ref names) = selection {
            let adapters = self
                .adapters
                .read()
                .map_err(|e| candle_core::Error::Msg(format!("Failed to acquire read lock: {e}")))?;
            for name in names {
                if !adapters.contains_key(name) {
                    return Err(candle_core::Error::Msg(format!(
                        "Adapter '{}' not registered",
                        name
                    )));
                }
            }
        }

        let mut sel = self
            .selection
            .write()
            .map_err(|e| candle_core::Error::Msg(format!("Failed to acquire write lock: {e}")))?;
        *sel = selection;
        Ok(())
    }

    /// Get the current selection strategy.
    pub fn get_selection(&self) -> Result<AdapterSelection> {
        let sel = self
            .selection
            .read()
            .map_err(|e| candle_core::Error::Msg(format!("Failed to acquire read lock: {e}")))?;
        Ok(sel.clone())
    }

    /// Get the consistent adapter ordering for XLoRA scalings.
    ///
    /// When using `AdapterSelection::Scalings`, scalings tensor dimension `n_adapters`
    /// corresponds to this ordering: `scalings[..., i]` weights `adapter_order()[i]`.
    pub fn adapter_order(&self) -> Result<Vec<String>> {
        let order = self
            .adapter_order
            .read()
            .map_err(|e| candle_core::Error::Msg(format!("Failed to acquire read lock: {e}")))?;
        Ok(order.clone())
    }

    /// Get the number of registered adapters.
    ///
    /// For XLoRA, this is the size of the `n_adapters` dimension in scalings.
    pub fn adapter_count(&self) -> Result<usize> {
        let order = self
            .adapter_order
            .read()
            .map_err(|e| candle_core::Error::Msg(format!("Failed to acquire read lock: {e}")))?;
        Ok(order.len())
    }

    /// Set which adapters are active for inference (convenience method).
    ///
    /// This is equivalent to `set_selection(AdapterSelection::Active(names))`.
    ///
    /// Adapters are applied in the order specified (first adapter's output
    /// is added first, etc.). All specified adapters must be registered.
    ///
    /// # Errors
    ///
    /// Returns an error if any adapter name is not registered.
    pub fn set_active(&self, names: &[impl AsRef<str>]) -> Result<()> {
        let name_vec: Vec<String> = names.iter().map(|s| s.as_ref().to_string()).collect();
        self.set_selection(AdapterSelection::Active(name_vec))
    }

    /// Get the currently active adapter names.
    ///
    /// Returns the names if selection is `Active`, or an empty vec for `Scalings`.
    pub fn get_active_names(&self) -> Result<Vec<String>> {
        let sel = self
            .selection
            .read()
            .map_err(|e| candle_core::Error::Msg(format!("Failed to acquire read lock: {e}")))?;
        match &*sel {
            AdapterSelection::Active(names) => Ok(names.clone()),
            AdapterSelection::Scalings(_) => Ok(Vec::new()),
        }
    }

    /// Get the number of currently active adapters.
    ///
    /// For `Active` selection, returns the count. For `Scalings`, returns 0
    /// (use `adapter_count()` for total registered adapters).
    pub fn active_count(&self) -> Result<usize> {
        let sel = self
            .selection
            .read()
            .map_err(|e| candle_core::Error::Msg(format!("Failed to acquire read lock: {e}")))?;
        match &*sel {
            AdapterSelection::Active(names) => Ok(names.len()),
            AdapterSelection::Scalings(_) => Ok(0),
        }
    }

    /// Set the default selection (convenience method for Active selection).
    ///
    /// This is used by [`restore_default`](Self::restore_default) to reset
    /// the selection after per-request overrides.
    pub fn set_default(&self, names: &[impl AsRef<str>]) -> Result<()> {
        let adapters = self
            .adapters
            .read()
            .map_err(|e| candle_core::Error::Msg(format!("Failed to acquire read lock: {e}")))?;

        // Validate all names exist
        for name in names {
            if !adapters.contains_key(name.as_ref()) {
                return Err(candle_core::Error::Msg(format!(
                    "Adapter '{}' not registered",
                    name.as_ref()
                )));
            }
        }
        drop(adapters);

        let name_vec: Vec<String> = names.iter().map(|s| s.as_ref().to_string()).collect();
        let mut default = self
            .default_selection
            .write()
            .map_err(|e| candle_core::Error::Msg(format!("Failed to acquire write lock: {e}")))?;
        *default = AdapterSelection::Active(name_vec);
        Ok(())
    }

    /// Set the default selection strategy.
    ///
    /// This is used by [`restore_default`](Self::restore_default) to reset
    /// the selection after per-request overrides.
    pub fn set_default_selection(&self, selection: AdapterSelection) -> Result<()> {
        // Validate Active selection
        if let AdapterSelection::Active(ref names) = selection {
            let adapters = self
                .adapters
                .read()
                .map_err(|e| candle_core::Error::Msg(format!("Failed to acquire read lock: {e}")))?;
            for name in names {
                if !adapters.contains_key(name) {
                    return Err(candle_core::Error::Msg(format!(
                        "Adapter '{}' not registered",
                        name
                    )));
                }
            }
        }

        let mut default = self
            .default_selection
            .write()
            .map_err(|e| candle_core::Error::Msg(format!("Failed to acquire write lock: {e}")))?;
        *default = selection;
        Ok(())
    }

    /// Restore the default selection.
    ///
    /// Call this after processing a request that used per-request adapter
    /// selection to reset to the default state.
    pub fn restore_default(&self) -> Result<()> {
        let default = self
            .default_selection
            .read()
            .map_err(|e| candle_core::Error::Msg(format!("Failed to acquire read lock: {e}")))?;
        let default_sel = default.clone();
        drop(default);

        let mut sel = self
            .selection
            .write()
            .map_err(|e| candle_core::Error::Msg(format!("Failed to acquire write lock: {e}")))?;
        *sel = default_sel;
        Ok(())
    }

    // ========================================================================
    // Weight Access
    // ========================================================================

    /// Get adapter weights for a specific layer based on current selection.
    ///
    /// For `Active` selection: Returns weights for active adapters.
    /// For `Scalings` selection: Returns weights for ALL adapters (caller applies scalings).
    ///
    /// Returns the A and B weight tensors along with the scale factor
    /// for each adapter that has weights for the given layer.
    pub fn get_active_weights_for_layer(
        &self,
        layer_idx: usize,
    ) -> Result<Vec<(Tensor, Tensor, f64)>> {
        let sel = self
            .selection
            .read()
            .map_err(|e| candle_core::Error::Msg(format!("Failed to acquire read lock: {e}")))?;

        match &*sel {
            AdapterSelection::Active(names) => {
                self.get_weights_for_names(layer_idx, names)
            }
            AdapterSelection::Scalings(_) => {
                // XLoRA mode: return all adapters in order
                let order = self
                    .adapter_order
                    .read()
                    .map_err(|e| candle_core::Error::Msg(format!("Failed to acquire read lock: {e}")))?;
                self.get_weights_for_names(layer_idx, &order)
            }
        }
    }

    /// Get weights for specific adapter names.
    fn get_weights_for_names(
        &self,
        layer_idx: usize,
        names: &[String],
    ) -> Result<Vec<(Tensor, Tensor, f64)>> {
        let adapters = self
            .adapters
            .read()
            .map_err(|e| candle_core::Error::Msg(format!("Failed to acquire read lock: {e}")))?;

        let mut result = Vec::with_capacity(names.len());
        for name in names.iter() {
            if let Some(adapter) = adapters.get(name) {
                if let Some((a, b)) = adapter.weights.get_layer(layer_idx) {
                    result.push((a.clone(), b.clone(), adapter.scale));
                }
            }
        }
        Ok(result)
    }

    /// Get ALL adapter weights for a layer, in adapter_order.
    ///
    /// Used by XLoRA to get weights for all adapters regardless of selection.
    /// The order matches [`adapter_order`](Self::adapter_order).
    pub fn get_all_weights_for_layer(
        &self,
        layer_idx: usize,
    ) -> Result<Vec<(Tensor, Tensor, f64)>> {
        let order = self
            .adapter_order
            .read()
            .map_err(|e| candle_core::Error::Msg(format!("Failed to acquire read lock: {e}")))?;
        self.get_weights_for_names(layer_idx, &order)
    }

    /// Get the current XLoRA scalings tensor, if in Scalings mode.
    pub fn get_scalings(&self) -> Result<Option<Tensor>> {
        let sel = self
            .selection
            .read()
            .map_err(|e| candle_core::Error::Msg(format!("Failed to acquire read lock: {e}")))?;
        match &*sel {
            AdapterSelection::Active(_) => Ok(None),
            AdapterSelection::Scalings(t) => Ok(Some(t.clone())),
        }
    }

    /// Check if currently using XLoRA scalings mode.
    pub fn is_scalings_mode(&self) -> Result<bool> {
        let sel = self
            .selection
            .read()
            .map_err(|e| candle_core::Error::Msg(format!("Failed to acquire read lock: {e}")))?;
        Ok(matches!(&*sel, AdapterSelection::Scalings(_)))
    }

    /// Offload an adapter's weights to CPU to save GPU memory.
    ///
    /// The adapter remains registered and can be re-loaded by calling
    /// [`ensure_loaded`](Self::ensure_loaded).
    pub fn offload(&self, name: &str) -> Result<()> {
        let mut adapters = self
            .adapters
            .write()
            .map_err(|e| candle_core::Error::Msg(format!("Failed to acquire write lock: {e}")))?;

        let adapter = adapters
            .get_mut(name)
            .ok_or_else(|| candle_core::Error::Msg(format!("Adapter '{name}' not registered")))?;

        adapter.weights.move_to_cpu()?;
        adapter.load_state = AdapterLoadState::Offloaded;
        Ok(())
    }

    /// Ensure an adapter's weights are loaded on the target device.
    ///
    /// If the adapter is offloaded, moves weights back to GPU.
    /// If the adapter is deferred, loads weights from disk.
    pub fn ensure_loaded(&self, name: &str) -> Result<()> {
        let mut adapters = self
            .adapters
            .write()
            .map_err(|e| candle_core::Error::Msg(format!("Failed to acquire write lock: {e}")))?;

        let adapter = adapters
            .get_mut(name)
            .ok_or_else(|| candle_core::Error::Msg(format!("Adapter '{name}' not registered")))?;

        match &adapter.load_state {
            AdapterLoadState::Ready => Ok(()),
            AdapterLoadState::Offloaded => {
                adapter.weights.move_to_device(&self.device)?;
                adapter.load_state = AdapterLoadState::Ready;
                Ok(())
            }
            AdapterLoadState::Deferred { path } => {
                // TODO: Implement deferred loading from path
                Err(candle_core::Error::Msg(format!(
                    "Deferred loading from {:?} not yet implemented",
                    path
                )))
            }
        }
    }

    /// Remove an adapter from the registry entirely.
    ///
    /// If the adapter is in the current selection (Active mode), it will be
    /// removed from the active list. Scalings mode is not affected.
    pub fn remove(&self, name: &str) -> Result<()> {
        // Remove from adapter_order first
        {
            let mut order = self.adapter_order.write().map_err(|e| {
                candle_core::Error::Msg(format!("Failed to acquire write lock: {e}"))
            })?;
            order.retain(|n| n != name);
        }

        // Remove from current selection if Active
        {
            let mut selection = self.selection.write().map_err(|e| {
                candle_core::Error::Msg(format!("Failed to acquire write lock: {e}"))
            })?;
            if let AdapterSelection::Active(ref mut names) = *selection {
                names.retain(|n| n != name);
            }
        }

        // Remove from default selection if Active
        {
            let mut default = self.default_selection.write().map_err(|e| {
                candle_core::Error::Msg(format!("Failed to acquire write lock: {e}"))
            })?;
            if let AdapterSelection::Active(ref mut names) = *default {
                names.retain(|n| n != name);
            }
        }

        // Remove from adapters
        let mut adapters = self
            .adapters
            .write()
            .map_err(|e| candle_core::Error::Msg(format!("Failed to acquire write lock: {e}")))?;
        adapters.remove(name);
        Ok(())
    }

    /// Check if an adapter is registered.
    pub fn contains(&self, name: &str) -> Result<bool> {
        let adapters = self
            .adapters
            .read()
            .map_err(|e| candle_core::Error::Msg(format!("Failed to acquire read lock: {e}")))?;
        Ok(adapters.contains_key(name))
    }

    /// Get all registered adapter names.
    pub fn list_adapters(&self) -> Result<Vec<String>> {
        let adapters = self
            .adapters
            .read()
            .map_err(|e| candle_core::Error::Msg(format!("Failed to acquire read lock: {e}")))?;
        Ok(adapters.keys().cloned().collect())
    }

    /// Get the device this registry manages adapters for.
    pub fn device(&self) -> &Device {
        &self.device
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn test_config() -> LoraConfig {
        LoraConfig {
            rank: 8,
            alpha: 16.0,
            target_modules: HashSet::from(["q_proj".to_string(), "v_proj".to_string()]),
        }
    }

    // Note: LoraConfig from mistralrs-quant doesn't have dropout field

    #[test]
    fn test_register_and_activate() {
        let registry = AdapterRegistry::new(Device::Cpu);

        // Register an adapter
        let weights = AdapterWeights::new();
        registry
            .register("test-adapter", test_config(), weights)
            .unwrap();

        // Check it's registered but not active
        assert!(registry.contains("test-adapter").unwrap());
        assert_eq!(registry.active_count().unwrap(), 0);

        // Activate it
        registry.set_active(&["test-adapter"]).unwrap();
        assert_eq!(registry.active_count().unwrap(), 1);
        assert_eq!(registry.get_active_names().unwrap(), vec!["test-adapter"]);
    }

    #[test]
    fn test_activate_unregistered_fails() {
        let registry = AdapterRegistry::new(Device::Cpu);

        let result = registry.set_active(&["nonexistent"]);
        assert!(result.is_err());
    }

    #[test]
    fn test_default_restore() {
        let registry = AdapterRegistry::new(Device::Cpu);

        // Register adapters
        registry
            .register("adapter-a", test_config(), AdapterWeights::new())
            .unwrap();
        registry
            .register("adapter-b", test_config(), AdapterWeights::new())
            .unwrap();

        // Set default
        registry.set_default(&["adapter-a"]).unwrap();
        registry.set_active(&["adapter-a"]).unwrap();

        // Override for a request
        registry.set_active(&["adapter-b"]).unwrap();
        assert_eq!(registry.get_active_names().unwrap(), vec!["adapter-b"]);

        // Restore default
        registry.restore_default().unwrap();
        assert_eq!(registry.get_active_names().unwrap(), vec!["adapter-a"]);
    }

    #[test]
    fn test_remove_adapter() {
        let registry = AdapterRegistry::new(Device::Cpu);

        registry
            .register("test", test_config(), AdapterWeights::new())
            .unwrap();
        registry.set_active(&["test"]).unwrap();

        // Remove should clear from active set too
        registry.remove("test").unwrap();
        assert!(!registry.contains("test").unwrap());
        assert_eq!(registry.active_count().unwrap(), 0);
    }

    #[test]
    fn test_adapter_order_consistency() {
        let registry = AdapterRegistry::new(Device::Cpu);

        // Register multiple adapters
        registry
            .register("adapter-a", test_config(), AdapterWeights::new())
            .unwrap();
        registry
            .register("adapter-b", test_config(), AdapterWeights::new())
            .unwrap();
        registry
            .register("adapter-c", test_config(), AdapterWeights::new())
            .unwrap();

        // Order should match registration order
        let order = registry.adapter_order().unwrap();
        assert_eq!(order, vec!["adapter-a", "adapter-b", "adapter-c"]);
        assert_eq!(registry.adapter_count().unwrap(), 3);

        // Re-registering shouldn't add duplicate
        registry
            .register("adapter-b", test_config(), AdapterWeights::new())
            .unwrap();
        assert_eq!(registry.adapter_count().unwrap(), 3);

        // Removing should update order
        registry.remove("adapter-b").unwrap();
        let order = registry.adapter_order().unwrap();
        assert_eq!(order, vec!["adapter-a", "adapter-c"]);
        assert_eq!(registry.adapter_count().unwrap(), 2);
    }

    #[test]
    fn test_scalings_mode() {
        let registry = AdapterRegistry::new(Device::Cpu);

        // Register adapters
        registry
            .register("adapter-a", test_config(), AdapterWeights::new())
            .unwrap();
        registry
            .register("adapter-b", test_config(), AdapterWeights::new())
            .unwrap();

        // Initially in Active mode
        assert!(!registry.is_scalings_mode().unwrap());

        // Create dummy scalings tensor [batch=1, seq=4, adapters=2]
        let scalings = Tensor::ones((1, 4, 2), candle_core::DType::F32, &Device::Cpu).unwrap();

        // Set Scalings mode
        registry
            .set_selection(AdapterSelection::Scalings(scalings.clone()))
            .unwrap();
        assert!(registry.is_scalings_mode().unwrap());

        // Get scalings back
        let retrieved = registry.get_scalings().unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().dims(), &[1, 4, 2]);

        // Restore to Active mode
        registry
            .set_selection(AdapterSelection::Active(vec!["adapter-a".into()]))
            .unwrap();
        assert!(!registry.is_scalings_mode().unwrap());
        assert!(registry.get_scalings().unwrap().is_none());
    }

    #[test]
    fn test_get_selection() {
        let registry = AdapterRegistry::new(Device::Cpu);

        registry
            .register("test", test_config(), AdapterWeights::new())
            .unwrap();

        // Default is Active([])
        let selection = registry.get_selection().unwrap();
        match selection {
            AdapterSelection::Active(names) => assert!(names.is_empty()),
            AdapterSelection::Scalings(_) => panic!("Expected Active"),
        }

        // Set active
        registry.set_active(&["test"]).unwrap();
        let selection = registry.get_selection().unwrap();
        match selection {
            AdapterSelection::Active(names) => assert_eq!(names, vec!["test"]),
            AdapterSelection::Scalings(_) => panic!("Expected Active"),
        }
    }
}
