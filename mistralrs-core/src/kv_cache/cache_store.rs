//! CacheStore trait and implementations for type-safe heterogeneous cache access.
//!
//! This module provides a unified interface for accessing different cache types
//! (KV, SSM, Sink) without runtime downcasting or panics.
//!
//! # Design
//!
//! Models call `cache.kv_mut(i)`, `cache.ssm_mut(i)`, etc. to access the cache
//! type they need for each layer. Methods return `Option`, allowing graceful
//! handling of mismatched cache types.
//!
//! # Example
//!
//! ```ignore
//! fn transform<C: CacheStore>(&self, hidden: Tensor, cache: &mut C) -> Result<Tensor> {
//!     for (i, layer) in self.layers.iter().enumerate() {
//!         let kv = cache.kv_mut(i).expect("attention layer requires KV cache");
//!         hidden = layer.forward(hidden, kv)?;
//!     }
//!     Ok(hidden)
//! }
//! ```

use candle_core::{DType, Device, Result, Tensor};

use super::{HybridCache, HybridCacheConfig, HybridLayerType, KvCache, MambaStatePool};

// =============================================================================
// SSM Cache (Mamba layers)
// =============================================================================

/// Configuration for SSM cache allocation.
#[derive(Debug, Clone)]
pub struct SsmCacheConfig {
    pub conv_dim: usize,
    pub d_conv: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    pub d_state: usize,
}

/// Cache state for State Space Model (Mamba) layers.
///
/// Unlike attention which caches K/V tensors, Mamba caches:
/// - Convolution state: rolling window for causal conv1d
/// - SSM state: recurrent state for selective state space
#[derive(Debug, Clone)]
pub struct SsmCache {
    /// Convolution state: [batch, conv_dim, d_conv]
    pub conv_state: Tensor,

    /// SSM recurrent state: [batch, n_heads, head_dim, d_state]
    pub ssm_state: Tensor,

    /// Current sequence position (for incremental decoding)
    pub seqlen_offset: usize,
}

impl SsmCache {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        batch_size: usize,
        conv_dim: usize,
        d_conv: usize,
        n_heads: usize,
        head_dim: usize,
        d_state: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let conv_state = Tensor::zeros((batch_size, conv_dim, d_conv), dtype, device)?;
        let ssm_state = Tensor::zeros((batch_size, n_heads, head_dim, d_state), dtype, device)?;
        Ok(Self {
            conv_state,
            ssm_state,
            seqlen_offset: 0,
        })
    }

    pub fn reset(&mut self) -> Result<()> {
        self.conv_state = self.conv_state.zeros_like()?;
        self.ssm_state = self.ssm_state.zeros_like()?;
        self.seqlen_offset = 0;
        Ok(())
    }
}

// =============================================================================
// Sink Cache (GPT-OSS / StreamingLLM)
// =============================================================================

/// Cache with attention sinks for infinite context streaming.
///
/// Attention sinks are special tokens (typically the first few) that absorb
/// excess attention probability, preventing degradation over long sequences.
#[derive(Debug, Clone)]
pub struct SinkCache {
    /// Standard KV cache for the sliding window
    pub kv: KvCache,

    /// Sink token KV (always retained, never evicted)
    pub sink_kv: Option<(Tensor, Tensor)>,

    /// Number of sink tokens to retain
    pub num_sink_tokens: usize,
}

impl SinkCache {
    pub fn new(kv: KvCache, num_sink_tokens: usize) -> Self {
        Self {
            kv,
            sink_kv: None,
            num_sink_tokens,
        }
    }

    pub fn reset(&mut self) {
        self.kv.reset();
        self.sink_kv = None;
    }

    pub fn current_seq_len(&self) -> usize {
        let sink_len = self
            .sink_kv
            .as_ref()
            .map(|(k, _)| k.dims()[2])
            .unwrap_or(0);
        sink_len + self.kv.current_seq_len()
    }
}

// =============================================================================
// CacheStore Trait
// =============================================================================

/// Trait for cache storage backends.
///
/// Models call `kv_mut()`, `ssm_mut()`, etc. to access the cache type they need.
/// Returns `Option` for type-safe access without panics.
///
/// # Implementing CacheStore
///
/// For simple KV-only models, implement `kv()` and `kv_mut()`. The SSM and sink
/// methods have default implementations returning `None`.
///
/// For hybrid models (Granite, Jamba), implement both KV and SSM methods.
pub trait CacheStore: Send + Sync {
    /// Number of layers in this cache store.
    fn num_layers(&self) -> usize;

    /// Get KV cache for layer `idx`. Returns None if layer doesn't use KV cache.
    fn kv(&self, idx: usize) -> Option<&KvCache>;

    /// Get mutable KV cache for layer `idx`.
    fn kv_mut(&mut self, idx: usize) -> Option<&mut KvCache>;

    /// Get SSM cache for layer `idx`. Returns None if layer doesn't use SSM cache.
    fn ssm(&self, _idx: usize) -> Option<&SsmCache> {
        None
    }

    /// Get mutable SSM cache for layer `idx`.
    fn ssm_mut(&mut self, _idx: usize) -> Option<&mut SsmCache> {
        None
    }

    /// Get sink cache for layer `idx`. Returns None if layer doesn't use sink cache.
    fn sink(&self, _idx: usize) -> Option<&SinkCache> {
        None
    }

    /// Get mutable sink cache for layer `idx`.
    fn sink_mut(&mut self, _idx: usize) -> Option<&mut SinkCache> {
        None
    }

    /// Get Mamba state pool for layer `idx` (continuous batching).
    ///
    /// Returns None for non-hybrid caches. Used by Mamba layers that need
    /// gather/scatter operations for batched inference.
    fn mamba_pool(&self, _idx: usize) -> Option<&MambaStatePool> {
        None
    }

    /// Get mutable Mamba state pool for layer `idx`.
    fn mamba_pool_mut(&mut self, _idx: usize) -> Option<&mut MambaStatePool> {
        None
    }

    /// Reset all caches (for new sequence).
    fn reset(&mut self);

    /// Current sequence length (from first non-empty cache).
    fn seq_len(&self) -> usize;
}

// =============================================================================
// CacheStore Implementation for Vec<KvCache>
// =============================================================================

impl CacheStore for Vec<KvCache> {
    fn num_layers(&self) -> usize {
        self.len()
    }

    fn kv(&self, idx: usize) -> Option<&KvCache> {
        self.get(idx)
    }

    fn kv_mut(&mut self, idx: usize) -> Option<&mut KvCache> {
        self.get_mut(idx)
    }

    fn reset(&mut self) {
        for cache in self.iter_mut() {
            cache.reset();
        }
    }

    fn seq_len(&self) -> usize {
        self.first().map(|c| c.current_seq_len()).unwrap_or(0)
    }
}

// Note: HybridCacheStore was removed - use HybridCache directly instead.
// HybridCache implements CacheStore and supports continuous batching via MambaStatePool.
// HybridLayerType is re-exported at the top of this file.

// =============================================================================
// SinkCacheStore
// =============================================================================

/// Cache storage with attention sinks (GPT-OSS / StreamingLLM).
#[derive(Debug)]
pub struct SinkCacheStore {
    caches: Vec<SinkCache>,
}

impl SinkCacheStore {
    pub fn new(num_layers: usize, kv_dim: usize, max_seq_len: usize, num_sink_tokens: usize) -> Self {
        let caches = (0..num_layers)
            .map(|_| {
                SinkCache::new(
                    KvCache::new_normal(kv_dim, max_seq_len, super::NormalCache::CACHE_GROW_SIZE),
                    num_sink_tokens,
                )
            })
            .collect();
        Self { caches }
    }
}

impl CacheStore for SinkCacheStore {
    fn num_layers(&self) -> usize {
        self.caches.len()
    }

    fn kv(&self, idx: usize) -> Option<&KvCache> {
        self.caches.get(idx).map(|c| &c.kv)
    }

    fn kv_mut(&mut self, idx: usize) -> Option<&mut KvCache> {
        self.caches.get_mut(idx).map(|c| &mut c.kv)
    }

    fn sink(&self, idx: usize) -> Option<&SinkCache> {
        self.caches.get(idx)
    }

    fn sink_mut(&mut self, idx: usize) -> Option<&mut SinkCache> {
        self.caches.get_mut(idx)
    }

    fn reset(&mut self) {
        for cache in &mut self.caches {
            cache.reset();
        }
    }

    fn seq_len(&self) -> usize {
        self.caches.first().map(|c| c.current_seq_len()).unwrap_or(0)
    }
}

// =============================================================================
// CacheLayout
// =============================================================================

/// Describes cache layout for a model.
/// Used by pipelines to allocate the correct cache store.
#[derive(Debug, Clone)]
pub enum CacheLayout {
    /// All layers use KV cache (Llama, Mistral, Qwen, etc.)
    UniformKv {
        num_layers: usize,
        dim: usize,
        max_seq_len: usize,
    },

    /// All layers use rotating KV cache (sliding window attention)
    UniformRotatingKv {
        num_layers: usize,
        dim: usize,
        window_size: usize,
    },

    /// Mixed layer types - attention and Mamba (Granite, Jamba, Falcon-H1)
    /// Uses HybridCache with MambaStatePool for continuous batching support.
    Hybrid {
        /// Layer types (Attention or Mamba for each layer)
        layer_types: Vec<HybridLayerType>,
        /// Maximum sequence length
        max_seq_len: usize,
        /// Maximum number of concurrent sequences (for Mamba state pool)
        max_num_seqs: usize,
        /// Mamba configuration
        ssm_config: SsmCacheConfig,
    },

    /// KV cache with attention sinks (GPT-OSS, StreamingLLM)
    WithSinks {
        num_layers: usize,
        dim: usize,
        max_seq_len: usize,
        num_sink_tokens: usize,
    },
}

impl CacheLayout {
    /// Create uniform KV cache layout (most common case).
    pub fn uniform_kv(num_layers: usize, dim: usize, max_seq_len: usize) -> Self {
        Self::UniformKv {
            num_layers,
            dim,
            max_seq_len,
        }
    }

    /// Create rotating KV cache layout for sliding window attention.
    pub fn sliding_window(num_layers: usize, dim: usize, window_size: usize) -> Self {
        Self::UniformRotatingKv {
            num_layers,
            dim,
            window_size,
        }
    }

    /// Create hybrid layout for mixed attention/Mamba models.
    ///
    /// # Arguments
    /// * `layer_types` - Layer types for each layer (Attention or Mamba)
    /// * `max_seq_len` - Maximum sequence length
    /// * `max_num_seqs` - Maximum concurrent sequences (for Mamba state pool)
    /// * `ssm_config` - Mamba/SSM configuration
    pub fn hybrid(
        layer_types: Vec<HybridLayerType>,
        max_seq_len: usize,
        max_num_seqs: usize,
        ssm_config: SsmCacheConfig,
    ) -> Self {
        Self::Hybrid {
            layer_types,
            max_seq_len,
            max_num_seqs,
            ssm_config,
        }
    }

    /// Create layout with attention sinks.
    pub fn with_sinks(num_layers: usize, dim: usize, max_seq_len: usize, num_sink_tokens: usize) -> Self {
        Self::WithSinks {
            num_layers,
            dim,
            max_seq_len,
            num_sink_tokens,
        }
    }

    /// Allocate cache store based on this layout.
    pub fn allocate(&self, dtype: DType, device: &Device) -> Result<Box<dyn CacheStore>> {
        match self {
            Self::UniformKv {
                num_layers,
                dim,
                max_seq_len,
            } => {
                let caches: Vec<KvCache> = (0..*num_layers)
                    .map(|_| KvCache::new_normal(*dim, *max_seq_len, super::NormalCache::CACHE_GROW_SIZE))
                    .collect();
                Ok(Box::new(caches))
            }
            Self::UniformRotatingKv {
                num_layers,
                dim,
                window_size,
            } => {
                let caches: Vec<KvCache> = (0..*num_layers)
                    .map(|_| KvCache::new_rotating(*dim, *window_size, super::NormalCache::CACHE_GROW_SIZE))
                    .collect();
                Ok(Box::new(caches))
            }
            Self::Hybrid {
                layer_types,
                max_seq_len,
                max_num_seqs,
                ssm_config,
            } => {
                let config = HybridCacheConfig {
                    layer_types: layer_types.clone(),
                    max_seq_len: *max_seq_len,
                    max_num_seqs: *max_num_seqs,
                    mamba_conv_dim: ssm_config.conv_dim,
                    mamba_d_conv: ssm_config.d_conv,
                    mamba_n_heads: ssm_config.n_heads,
                    mamba_head_dim: ssm_config.head_dim,
                    mamba_d_state: ssm_config.d_state,
                };
                Ok(Box::new(HybridCache::new(config, dtype, device)?))
            }
            Self::WithSinks {
                num_layers,
                dim,
                max_seq_len,
                num_sink_tokens,
            } => Ok(Box::new(SinkCacheStore::new(
                *num_layers,
                *dim,
                *max_seq_len,
                *num_sink_tokens,
            ))),
        }
    }
}
