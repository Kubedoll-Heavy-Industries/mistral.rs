//! XLoRA classifier for computing per-token adapter scalings.
//!
//! The classifier is an MLP that takes hidden states from a model's last layer
//! and produces scalings that weight each adapter's contribution per token.
//!
//! # Architecture
//!
//! ```text
//! hidden_states [batch, seq, hidden_dim]
//!         │
//!         ▼
//! ┌───────────────┐
//! │  Inner Layers │  (0+ layers based on xlora_depth)
//! │  Linear/ReLU  │
//! │  /Dropout     │
//! └───────┬───────┘
//!         │
//!         ▼
//! ┌───────────────┐
//! │  Last Linear  │  → [batch, seq, n_classes * n_layers] or [batch, seq, n_classes]
//! └───────┬───────┘
//!         │
//!         ▼
//! ┌───────────────┐
//! │  Reshape      │  → [batch, seq, n_layers, n_classes]
//! └───────┬───────┘
//!         │
//!         ▼
//! ┌───────────────┐
//! │  Softmax      │  (optional, temperature-scaled)
//! └───────┬───────┘
//!         │
//!         ▼
//! ┌───────────────┐
//! │  Top-K        │  (optional sparse selection)
//! └───────┬───────┘
//!         │
//!         ▼
//! scalings [batch, seq, n_layers, n_classes]
//! ```

use crate::layers::{linear, linear_no_bias};
use crate::ops::{TopKLastDimOp, TopKOutput};
use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{activation, ops::softmax_last_dim, Dropout, Linear, Module, ModuleT};
use mistralrs_quant::ShardedVarBuilder;

use super::XLoraConfig;

/// Temperature-scaled softmax for normalizing scalings.
#[derive(Debug)]
struct TemperatureScaledSoftmax {
    temp: f64,
}

impl Module for TemperatureScaledSoftmax {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        softmax_last_dim(&(xs / self.temp)?)
    }
}

/// XLoRA classifier that computes per-token adapter scalings from hidden states.
///
/// This is an optional component used for XLoRA inference. When present,
/// the model performs a two-pass forward:
/// 1. **Scaling pass**: Forward with dummy scalings to get hidden states
/// 2. **Classifier**: This struct computes real scalings from hidden states
/// 3. **Main pass**: Forward with real scalings applied to adapters
///
/// # Usage
///
/// ```ignore
/// let classifier = XLoraClassifier::new(config, n_layers, n_adapters, vb, false)?;
///
/// // Get hidden states from model (scaling pass)
/// let hidden_states = model.forward_hidden(&input)?;
///
/// // Compute scalings
/// let scalings = classifier.forward(hidden_states)?;
///
/// // Set scalings on registry
/// registry.set_selection(AdapterSelection::Scalings(scalings));
///
/// // Forward with real scalings
/// let output = model.forward(&input)?;
/// ```
pub struct XLoraClassifier {
    /// Final output layer projecting to n_classes (or n_classes * n_layers).
    last: Linear,
    /// Hidden layers (empty for depth=1).
    inner: Vec<Box<dyn ModuleT + Send + Sync>>,
    /// Optional temperature-scaled softmax.
    softmax: Option<TemperatureScaledSoftmax>,
    /// Value used for dummy scalings during scaling pass.
    scaling_pass_value: f64,
    /// Number of model layers (for layerwise scalings expansion).
    model_layers: usize,
    /// Number of adapter classes.
    n_classes: usize,
    /// Full configuration.
    pub config: XLoraConfig,
}

impl std::fmt::Debug for XLoraClassifier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("XLoraClassifier")
            .field("model_layers", &self.model_layers)
            .field("n_classes", &self.n_classes)
            .field("scaling_pass_value", &self.scaling_pass_value)
            .field("has_softmax", &self.softmax.is_some())
            .field("inner_layers", &self.inner.len())
            .finish()
    }
}

impl XLoraClassifier {
    /// Create a new XLoRA classifier.
    ///
    /// # Arguments
    ///
    /// * `config` - XLoRA configuration
    /// * `n_layers` - Number of model layers (for layerwise scalings)
    /// * `n_classes` - Number of adapter classes (adapters)
    /// * `vb` - Variable builder for loading weights
    /// * `is_quantized` - If true, convert weights to F32
    pub fn new(
        config: XLoraConfig,
        n_layers: usize,
        n_classes: usize,
        vb: ShardedVarBuilder,
        is_quantized: bool,
    ) -> Result<Self> {
        if config.enable_softmax_topk {
            candle_core::bail!("`enable_softmax_topk` is not implemented");
        }

        let (last, inner) = Self::build_layers(&config, n_layers, n_classes, &vb, is_quantized)?;

        let last = if is_quantized {
            Linear::new(
                last.weight().to_dtype(DType::F32)?,
                last.bias().map(|x| x.to_dtype(DType::F32).unwrap()),
            )
        } else {
            last
        };

        Ok(Self {
            last,
            inner,
            softmax: if config.enable_softmax {
                Some(TemperatureScaledSoftmax {
                    temp: config.softmax_temperature,
                })
            } else {
                None
            },
            scaling_pass_value: config.scaling_pass_value,
            model_layers: n_layers,
            n_classes,
            config,
        })
    }

    /// Build the classifier layers based on depth configuration.
    fn build_layers(
        config: &XLoraConfig,
        n_layers: usize,
        n_classes: usize,
        vb: &ShardedVarBuilder,
        is_quantized: bool,
    ) -> Result<(Linear, Vec<Box<dyn ModuleT + Send + Sync>>)> {
        let output_dim = if config.layerwise_scalings {
            n_classes * n_layers
        } else {
            n_classes
        };

        match config.xlora_depth {
            1 => Self::build_depth_1(config, output_dim, vb, is_quantized),
            2 => Self::build_depth_2(config, output_dim, vb, is_quantized),
            _ => Self::build_depth_n(config, output_dim, vb, is_quantized),
        }
    }

    /// Build single-layer classifier (depth=1).
    fn build_depth_1(
        config: &XLoraConfig,
        output_dim: usize,
        vb: &ShardedVarBuilder,
        is_quantized: bool,
    ) -> Result<(Linear, Vec<Box<dyn ModuleT + Send + Sync>>)> {
        assert!(vb.contains_tensor("last.weight"));
        let lin = if config.use_bias {
            assert!(vb.contains_tensor("last.bias"));
            linear(config.hidden_size, output_dim, vb.pp("last"))?
        } else {
            linear_no_bias(config.hidden_size, output_dim, vb.pp("last"))?
        };

        let lin = if is_quantized {
            Linear::new(
                lin.weight().to_dtype(DType::F32)?,
                lin.bias().map(|x| x.to_dtype(DType::F32).unwrap()),
            )
        } else {
            lin
        };

        Ok((lin, vec![]))
    }

    /// Build two-layer classifier (depth=2).
    fn build_depth_2(
        config: &XLoraConfig,
        output_dim: usize,
        vb: &ShardedVarBuilder,
        is_quantized: bool,
    ) -> Result<(Linear, Vec<Box<dyn ModuleT + Send + Sync>>)> {
        let mut inner: Vec<Box<dyn ModuleT + Send + Sync>> = Vec::new();

        // First hidden layer
        assert!(vb.contains_tensor("inner.0.weight"));
        let lin = if config.use_bias {
            assert!(vb.contains_tensor("inner.0.bias"));
            linear(config.hidden_size, config.xlora_size, vb.pp("inner.0"))?
        } else {
            linear_no_bias(config.hidden_size, config.xlora_size, vb.pp("inner.0"))?
        };

        inner.push(Box::new(if is_quantized {
            Linear::new(
                lin.weight().to_dtype(DType::F32)?,
                lin.bias().map(|x| x.to_dtype(DType::F32).unwrap()),
            )
        } else {
            lin
        }));

        if config.enable_relu_and_dropout {
            inner.push(Box::new(activation::Activation::Relu));
            inner.push(Box::new(Dropout::new(config.xlora_dropout_p)));
        }

        // Output layer
        assert!(vb.contains_tensor("last.weight"));
        let last = if config.use_bias {
            assert!(vb.contains_tensor("last.bias"));
            linear(config.xlora_size, output_dim, vb.pp("last"))?
        } else {
            linear_no_bias(config.xlora_size, output_dim, vb.pp("last"))?
        };

        let last = if is_quantized {
            Linear::new(
                last.weight().to_dtype(DType::F32)?,
                last.bias().map(|x| x.to_dtype(DType::F32).unwrap()),
            )
        } else {
            last
        };

        Ok((last, inner))
    }

    /// Build multi-layer classifier (depth≥3).
    fn build_depth_n(
        config: &XLoraConfig,
        output_dim: usize,
        vb: &ShardedVarBuilder,
        is_quantized: bool,
    ) -> Result<(Linear, Vec<Box<dyn ModuleT + Send + Sync>>)> {
        let mut inner: Vec<Box<dyn ModuleT + Send + Sync>> = Vec::new();

        // First hidden layer: hidden_size → xlora_size
        assert!(vb.contains_tensor("inner.0.weight"));
        let lin = if config.use_bias {
            assert!(vb.contains_tensor("inner.0.bias"));
            linear(config.hidden_size, config.xlora_size, vb.pp("inner.0"))?
        } else {
            linear_no_bias(config.hidden_size, config.xlora_size, vb.pp("inner.0"))?
        };

        inner.push(Box::new(if is_quantized {
            Linear::new(
                lin.weight().to_dtype(DType::F32)?,
                lin.bias().map(|x| x.to_dtype(DType::F32).unwrap()),
            )
        } else {
            lin
        }));

        if config.enable_relu_and_dropout {
            inner.push(Box::new(activation::Activation::Relu));
            inner.push(Box::new(Dropout::new(config.xlora_dropout_p)));
        }

        // Middle hidden layers: xlora_size → xlora_size
        for i in 1..=config.xlora_depth - 2 {
            assert!(vb.contains_tensor(&format!("inner.{i}.weight")));
            let lin = if config.use_bias {
                assert!(vb.contains_tensor(&format!("inner.{i}.bias")));
                linear(config.xlora_size, config.xlora_size, vb.pp(format!("inner.{i}")))?
            } else {
                linear_no_bias(config.xlora_size, config.xlora_size, vb.pp(format!("inner.{i}")))?
            };

            inner.push(Box::new(if is_quantized {
                Linear::new(
                    lin.weight().to_dtype(DType::F32)?,
                    lin.bias().map(|x| x.to_dtype(DType::F32).unwrap()),
                )
            } else {
                lin
            }));

            if config.enable_relu_and_dropout {
                inner.push(Box::new(activation::Activation::Relu));
                inner.push(Box::new(Dropout::new(config.xlora_dropout_p)));
            }
        }

        // Output layer: xlora_size → output_dim
        assert!(vb.contains_tensor("last.weight"));
        let last = if config.use_bias {
            assert!(vb.contains_tensor("last.bias"));
            linear(config.xlora_size, output_dim, vb.pp("last"))?
        } else {
            linear_no_bias(config.xlora_size, output_dim, vb.pp("last"))?
        };

        let last = if is_quantized {
            Linear::new(
                last.weight().to_dtype(DType::F32)?,
                last.bias().map(|x| x.to_dtype(DType::F32).unwrap()),
            )
        } else {
            last
        };

        Ok((last, inner))
    }

    /// Compute adapter scalings from hidden states.
    ///
    /// # Arguments
    ///
    /// * `hidden_states` - Hidden states from model's last layer.
    ///   Shape: `[batch, seq_len, hidden_dim]`
    ///
    /// # Returns
    ///
    /// Scalings tensor of shape `[batch, seq_len, n_layers, n_classes]`
    /// where each value is the weight for that adapter on that layer/token.
    pub fn forward(&self, mut hidden_states: Tensor) -> Result<Tensor> {
        // Pass through inner layers
        for layer in &self.inner {
            hidden_states = layer.forward_t(&hidden_states, true)?;
        }

        // Final projection to logits
        let mut logits = self.last.forward(&hidden_states)?;

        // Expand to layerwise if not already
        if !self.config.layerwise_scalings {
            logits = logits.unsqueeze(2)?;
            logits = logits.expand((
                logits.dims()[0],
                logits.dims()[1],
                self.model_layers,
                logits.dims()[3],
            ))?;
        }

        // Reshape to [batch, seq, n_layers, n_classes]
        let mut scalings = logits.reshape((
            logits.dims()[0],
            logits.dims()[1],
            self.model_layers,
            self.n_classes,
        ))?;

        // Optional softmax normalization
        if let Some(ref softmax) = self.softmax {
            scalings = softmax.forward(&scalings)?;
        }

        // Optional top-k filtering
        let scalings = if let Some(topk_lora) = self.config.top_k_lora {
            let TopKOutput { values: _, indices } = scalings.topk(topk_lora)?;
            let scalings_zeroed = scalings.zeros_like()?;
            scalings_zeroed.scatter_add(
                &indices,
                &scalings.gather(&indices, D::Minus1)?,
                D::Minus1,
            )?
        } else {
            scalings
        };

        Ok(scalings)
    }

    /// Get dummy scalings for the scaling pass.
    ///
    /// During the scaling pass, adapters should have minimal effect so we can
    /// collect clean hidden states. This returns a tensor filled with
    /// `scaling_pass_value` (typically 0).
    pub fn get_dummy_scalings(
        &self,
        bs: usize,
        seq_len: usize,
        device: &Device,
        dtype: DType,
    ) -> Result<Tensor> {
        #[allow(clippy::cast_possible_truncation)]
        Tensor::full(
            self.scaling_pass_value as f32,
            (bs, seq_len, self.model_layers, self.n_classes),
            device,
        )?
        .to_dtype(dtype)
    }

    /// Get the global scaling weight from configuration.
    pub fn get_global_scaling_weight(&self) -> f64 {
        self.config.global_scaling_weight
    }

    /// Get the number of model layers.
    pub fn model_layers(&self) -> usize {
        self.model_layers
    }

    /// Get the number of adapter classes.
    pub fn n_classes(&self) -> usize {
        self.n_classes
    }
}
