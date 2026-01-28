//! XLoRA configuration for the classifier network.
//!
//! This is used to configure the XLoRA classifier that computes per-token
//! adapter scalings from hidden states.

use std::collections::HashMap;

use either::Either;
use serde::Deserialize;

fn true_default() -> bool {
    true
}

fn false_default() -> bool {
    false
}

fn default_1() -> usize {
    1
}

fn default_2048() -> usize {
    2048
}

fn default_dropout() -> f32 {
    0.2
}

fn default_1f64() -> f64 {
    1.0
}

fn default_0f64() -> f64 {
    0.0
}

/// Configuration for the XLoRA classifier network.
///
/// The classifier is an MLP that takes hidden states from the model's last layer
/// and produces per-token scalings for each adapter.
///
/// # Architecture
///
/// The depth parameter controls the network structure:
/// - `xlora_depth=1`: Single linear layer (hidden_size → output_dim)
/// - `xlora_depth=2`: Two-layer MLP with optional ReLU+Dropout
/// - `xlora_depth≥3`: Multi-layer MLP with `xlora_size` hidden units
///
/// # Output Shape
///
/// - Global scalings: `[batch, seq_len, n_adapters]`
/// - Layerwise scalings: `[batch, seq_len, n_layers, n_adapters]`
#[derive(Clone, Debug, Deserialize)]
pub struct XLoraConfig {
    /// Input hidden size (must match model's hidden dimension).
    pub hidden_size: usize,

    /// Base model identifier (for reference).
    pub base_model_id: String,

    /// Adapter names or name-to-path mapping.
    /// Used during loading to identify which adapters to load.
    #[serde(rename = "adapters")]
    #[serde(with = "either::serde_untagged")]
    pub adapters: Either<Vec<String>, HashMap<String, String>>,

    /// If true, each model layer gets different adapter weights.
    /// Output shape: `[batch, seq, n_layers, n_adapters]`
    /// If false, same weights across all layers.
    /// Output shape: `[batch, seq, n_adapters]` (expanded to 4D internally)
    #[serde(default = "false_default")]
    pub layerwise_scalings: bool,

    /// Enable ReLU activation and dropout between hidden layers.
    #[serde(default = "false_default")]
    pub enable_relu_and_dropout: bool,

    /// Number of layers in the classifier MLP.
    /// 1 = single linear, 2+ = multi-layer with hidden units.
    #[serde(default = "default_1")]
    pub xlora_depth: usize,

    /// Hidden layer size for depth > 1.
    #[serde(default = "default_2048")]
    pub xlora_size: usize,

    /// Dropout probability for hidden layers.
    #[serde(default = "default_dropout")]
    pub xlora_dropout_p: f32,

    /// Apply temperature-scaled softmax to output scalings.
    /// Ensures scalings sum to 1 per token.
    #[serde(default = "true_default")]
    pub enable_softmax: bool,

    /// Temperature for softmax (lower = sharper distribution).
    #[serde(default = "default_1f64")]
    pub softmax_temperature: f64,

    /// Value used for dummy scalings during the scaling pass.
    /// Set to 0 to effectively disable adapters during hidden state collection.
    #[serde(default = "default_0f64")]
    pub scaling_pass_value: f64,

    /// Whether the original model used trainable adapters (metadata only).
    #[serde(default = "false_default", rename = "use_trainable_adapters")]
    pub _use_trainable_adapters: bool,

    /// Use bias in classifier linear layers.
    #[serde(default = "true_default")]
    pub use_bias: bool,

    /// Global weight multiplier for all adapter outputs.
    #[serde(default = "default_1f64")]
    pub global_scaling_weight: f64,

    /// Keep only top-k adapters per token (sparse selection).
    /// None = use all adapters.
    pub top_k_lora: Option<usize>,

    /// Not implemented.
    #[serde(default = "false_default")]
    pub enable_softmax_topk: bool,
}
