//! Thread-safe, model-scoped adapter registry for runtime adapter management.
//!
//! This module provides [`AdapterRegistry`], which replaces the thread-local
//! LoRA registry with a thread-safe, model-scoped solution that enables:
//!
//! - Runtime adapter switching without model reload
//! - Per-request adapter selection
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
//! // Activate adapters for inference
//! registry.set_active(&["style-formal", "skill-coding"])?;
//!
//! // Run inference with active adapters...
//!
//! // Switch to different adapter set
//! registry.set_active(&["style-casual"])?;
//! ```

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::RwLock;

use candle_core::{Device, Result, Tensor};

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
        match (self.a_weights.get(&layer_idx), self.b_weights.get(&layer_idx)) {
            (Some(a), Some(b)) => Some((a, b)),
            _ => None,
        }
    }

    /// Move all weights to CPU.
    pub fn to_cpu(&mut self) -> Result<()> {
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
    pub fn to_device(&mut self, device: &Device) -> Result<()> {
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
/// - Activation (selecting which adapters to use)
/// - Offloading (moving weights to CPU to save GPU memory)
/// - Removal (unloading adapters entirely)
///
/// # Thread Safety
///
/// All operations are protected by `RwLock`, allowing concurrent reads
/// (e.g., multiple inference threads reading active adapters) with
/// exclusive writes (e.g., switching active adapter set).
#[derive(Debug)]
pub struct AdapterRegistry {
    /// All registered adapters by name.
    adapters: RwLock<HashMap<String, LoadedAdapter>>,
    /// Currently active adapter names, in stacking order.
    active: RwLock<Vec<String>>,
    /// Default adapter set to restore after per-request overrides.
    default_active: RwLock<Vec<String>>,
    /// Device for loading/moving weights.
    device: Device,
}

impl AdapterRegistry {
    /// Create a new adapter registry for the given device.
    pub fn new(device: Device) -> Self {
        Self {
            adapters: RwLock::new(HashMap::new()),
            active: RwLock::new(Vec::new()),
            default_active: RwLock::new(Vec::new()),
            device,
        }
    }

    /// Register an adapter with pre-loaded weights.
    ///
    /// The adapter is added to the registry but not activated.
    /// Call [`set_active`](Self::set_active) to activate it.
    pub fn register(
        &self,
        name: impl Into<String>,
        config: LoraConfig,
        weights: AdapterWeights,
    ) -> Result<()> {
        let name = name.into();
        let adapter = LoadedAdapter::new(config, weights);
        let mut adapters = self.adapters.write().map_err(|e| {
            candle_core::Error::Msg(format!("Failed to acquire write lock: {e}"))
        })?;
        adapters.insert(name, adapter);
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
        let mut adapters = self.adapters.write().map_err(|e| {
            candle_core::Error::Msg(format!("Failed to acquire write lock: {e}"))
        })?;
        adapters.insert(name, adapter);
        Ok(())
    }

    /// Set which adapters are active for inference.
    ///
    /// Adapters are applied in the order specified (first adapter's output
    /// is added first, etc.). All specified adapters must be registered.
    ///
    /// # Errors
    ///
    /// Returns an error if any adapter name is not registered.
    pub fn set_active(&self, names: &[impl AsRef<str>]) -> Result<()> {
        let adapters = self.adapters.read().map_err(|e| {
            candle_core::Error::Msg(format!("Failed to acquire read lock: {e}"))
        })?;

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

        // Update active set
        let mut active = self.active.write().map_err(|e| {
            candle_core::Error::Msg(format!("Failed to acquire write lock: {e}"))
        })?;
        active.clear();
        active.extend(names.iter().map(|s| s.as_ref().to_string()));
        Ok(())
    }

    /// Get the currently active adapter names.
    pub fn get_active_names(&self) -> Result<Vec<String>> {
        let active = self.active.read().map_err(|e| {
            candle_core::Error::Msg(format!("Failed to acquire read lock: {e}"))
        })?;
        Ok(active.clone())
    }

    /// Get the number of currently active adapters.
    pub fn active_count(&self) -> Result<usize> {
        let active = self.active.read().map_err(|e| {
            candle_core::Error::Msg(format!("Failed to acquire read lock: {e}"))
        })?;
        Ok(active.len())
    }

    /// Set the default adapter set.
    ///
    /// This is used by [`restore_default`](Self::restore_default) to reset
    /// the active adapters after per-request overrides.
    pub fn set_default(&self, names: &[impl AsRef<str>]) -> Result<()> {
        let adapters = self.adapters.read().map_err(|e| {
            candle_core::Error::Msg(format!("Failed to acquire read lock: {e}"))
        })?;

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

        let mut default = self.default_active.write().map_err(|e| {
            candle_core::Error::Msg(format!("Failed to acquire write lock: {e}"))
        })?;
        default.clear();
        default.extend(names.iter().map(|s| s.as_ref().to_string()));
        Ok(())
    }

    /// Restore the default adapter set.
    ///
    /// Call this after processing a request that used per-request adapter
    /// selection to reset to the default state.
    pub fn restore_default(&self) -> Result<()> {
        let default = self.default_active.read().map_err(|e| {
            candle_core::Error::Msg(format!("Failed to acquire read lock: {e}"))
        })?;
        let default_names: Vec<String> = default.clone();
        drop(default);

        let mut active = self.active.write().map_err(|e| {
            candle_core::Error::Msg(format!("Failed to acquire write lock: {e}"))
        })?;
        *active = default_names;
        Ok(())
    }

    /// Get adapter weights for a specific layer.
    ///
    /// Returns the A and B weight tensors along with the scale factor
    /// for each active adapter that has weights for the given layer.
    pub fn get_active_weights_for_layer(
        &self,
        layer_idx: usize,
    ) -> Result<Vec<(Tensor, Tensor, f64)>> {
        let active = self.active.read().map_err(|e| {
            candle_core::Error::Msg(format!("Failed to acquire read lock: {e}"))
        })?;
        let adapters = self.adapters.read().map_err(|e| {
            candle_core::Error::Msg(format!("Failed to acquire read lock: {e}"))
        })?;

        let mut result = Vec::with_capacity(active.len());
        for name in active.iter() {
            if let Some(adapter) = adapters.get(name) {
                if let Some((a, b)) = adapter.weights.get_layer(layer_idx) {
                    result.push((a.clone(), b.clone(), adapter.scale));
                }
            }
        }
        Ok(result)
    }

    /// Offload an adapter's weights to CPU to save GPU memory.
    ///
    /// The adapter remains registered and can be re-loaded by calling
    /// [`ensure_loaded`](Self::ensure_loaded).
    pub fn offload(&self, name: &str) -> Result<()> {
        let mut adapters = self.adapters.write().map_err(|e| {
            candle_core::Error::Msg(format!("Failed to acquire write lock: {e}"))
        })?;

        let adapter = adapters.get_mut(name).ok_or_else(|| {
            candle_core::Error::Msg(format!("Adapter '{name}' not registered"))
        })?;

        adapter.weights.to_cpu()?;
        adapter.load_state = AdapterLoadState::Offloaded;
        Ok(())
    }

    /// Ensure an adapter's weights are loaded on the target device.
    ///
    /// If the adapter is offloaded, moves weights back to GPU.
    /// If the adapter is deferred, loads weights from disk.
    pub fn ensure_loaded(&self, name: &str) -> Result<()> {
        let mut adapters = self.adapters.write().map_err(|e| {
            candle_core::Error::Msg(format!("Failed to acquire write lock: {e}"))
        })?;

        let adapter = adapters.get_mut(name).ok_or_else(|| {
            candle_core::Error::Msg(format!("Adapter '{name}' not registered"))
        })?;

        match &adapter.load_state {
            AdapterLoadState::Ready => Ok(()),
            AdapterLoadState::Offloaded => {
                adapter.weights.to_device(&self.device)?;
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
    /// If the adapter is currently active, it will be removed from the
    /// active set as well.
    pub fn remove(&self, name: &str) -> Result<()> {
        // Remove from active set first
        {
            let mut active = self.active.write().map_err(|e| {
                candle_core::Error::Msg(format!("Failed to acquire write lock: {e}"))
            })?;
            active.retain(|n| n != name);
        }

        // Remove from default set
        {
            let mut default = self.default_active.write().map_err(|e| {
                candle_core::Error::Msg(format!("Failed to acquire write lock: {e}"))
            })?;
            default.retain(|n| n != name);
        }

        // Remove from adapters
        let mut adapters = self.adapters.write().map_err(|e| {
            candle_core::Error::Msg(format!("Failed to acquire write lock: {e}"))
        })?;
        adapters.remove(name);
        Ok(())
    }

    /// Check if an adapter is registered.
    pub fn contains(&self, name: &str) -> Result<bool> {
        let adapters = self.adapters.read().map_err(|e| {
            candle_core::Error::Msg(format!("Failed to acquire read lock: {e}"))
        })?;
        Ok(adapters.contains_key(name))
    }

    /// Get all registered adapter names.
    pub fn list_adapters(&self) -> Result<Vec<String>> {
        let adapters = self.adapters.read().map_err(|e| {
            candle_core::Error::Msg(format!("Failed to acquire read lock: {e}"))
        })?;
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
        registry.register("test-adapter", test_config(), weights).unwrap();

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
        registry.register("adapter-a", test_config(), AdapterWeights::new()).unwrap();
        registry.register("adapter-b", test_config(), AdapterWeights::new()).unwrap();

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

        registry.register("test", test_config(), AdapterWeights::new()).unwrap();
        registry.set_active(&["test"]).unwrap();

        // Remove should clear from active set too
        registry.remove("test").unwrap();
        assert!(!registry.contains("test").unwrap());
        assert_eq!(registry.active_count().unwrap(), 0);
    }
}
