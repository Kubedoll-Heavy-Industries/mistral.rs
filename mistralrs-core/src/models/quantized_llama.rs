#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
#![allow(deprecated)] // Uses deprecated traits during migration

use std::collections::HashMap;
use std::sync::Arc;

use candle_core::quantized::ggml_file;
use candle_core::quantized::QTensor;
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Embedding, Module};
use mistralrs_quant::{GgufMatMul, QuantMethod, QuantMethodConfig};

use crate::attention::SdpaParams;
use crate::device_map::DeviceMapper;
use crate::gguf::Content;
use crate::layers::{
    reshape_attn_output, reshape_for_attn, sdpa_with_cache, Activation, CausalMasker, MatMul, Mlp,
    RmsNorm, RotaryEmbedding,
};
use crate::layers_masker::PastKvLenCache;
use crate::models::{LanguageModel, Model, TransformContext, TransformerModel};
use crate::paged_attention::{AttentionImplementation, PagedAttention};
use crate::pipeline::text_models_inputs_processor::PagedAttentionInputMetadata;
use crate::pipeline::KvCache;
use crate::utils::gguf_metadata::ContentMetadata;
use crate::utils::model_config as ModelConfig;
use crate::utils::progress::{new_multi_progress, NiceProgressBar};
// Default fallback for models that don't specify context_length
const DEFAULT_MAX_SEQ_LEN: u32 = 4096;

enum MlpOrMoe {
    Mlp(Mlp),
    MoE {
        n_expert_used: usize,
        feed_forward_gate_inp: Arc<dyn QuantMethod>,
        experts: Vec<Mlp>,
    },
}

impl MlpOrMoe {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::MoE {
                feed_forward_gate_inp,
                experts,
                n_expert_used,
            } => {
                let (b_size, seq_len, hidden_dim) = xs.dims3()?;
                let xs = xs.reshape(((), hidden_dim))?;
                let router_logits = MatMul.qmethod_matmul(&xs, &**feed_forward_gate_inp)?;
                let routing_weights = candle_nn::ops::softmax_last_dim(&router_logits)?;

                // In order to extract topk, we extract the data from the tensor and manipulate it
                // directly. Maybe we will want to use some custom ops instead at some point.
                let routing_weights = routing_weights.to_dtype(DType::F32)?.to_vec2::<f32>()?;

                // routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
                // top_x contains the row indexes to evaluate for each expert.
                let mut top_x = vec![vec![]; experts.len()];
                let mut selected_rws = vec![vec![]; experts.len()];
                for (row_idx, rw) in routing_weights.iter().enumerate() {
                    let mut dst = (0..rw.len() as u32).collect::<Vec<u32>>();
                    dst.sort_by(|&i, &j| rw[j as usize].total_cmp(&rw[i as usize]));
                    let mut sum_routing_weights = 0f32;
                    for &expert_idx in dst.iter().take(*n_expert_used) {
                        let expert_idx = expert_idx as usize;
                        let routing_weight = rw[expert_idx];
                        sum_routing_weights += routing_weight;
                        top_x[expert_idx].push(row_idx as u32);
                    }
                    for &expert_idx in dst.iter().take(*n_expert_used) {
                        let expert_idx = expert_idx as usize;
                        let routing_weight = rw[expert_idx];
                        selected_rws[expert_idx].push(routing_weight / sum_routing_weights)
                    }
                }

                // routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
                // expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

                let mut ys = xs.zeros_like()?;
                for (expert_idx, expert_layer) in experts.iter().enumerate() {
                    let top_x = &top_x[expert_idx];
                    if top_x.is_empty() {
                        continue;
                    }
                    let top_x = Tensor::new(top_x.as_slice(), xs.device())?;
                    let selected_rws =
                        Tensor::new(selected_rws[expert_idx].as_slice(), xs.device())?
                            .reshape(((), 1))?;
                    // Index the correct hidden states and compute the expert hidden state for
                    // the current expert. We need to make sure to multiply the output hidden
                    // states by `routing_weights` on the corresponding tokens (top-1 and top-2)
                    let current_state = xs.index_select(&top_x, 0)?.reshape(((), hidden_dim))?;
                    // current_hidden_states = expert_layer(current_state, routing_weights[top_x_list, idx_list, None])
                    let current_hidden_states = expert_layer.forward(&current_state)?;
                    let current_hidden_states =
                        current_hidden_states.broadcast_mul(&selected_rws)?;
                    ys = ys.index_add(&top_x, &current_hidden_states, 0)?;
                }

                let ys = ys.reshape((b_size, seq_len, hidden_dim))?;
                Ok(ys)
            }
            Self::Mlp(mlp) => mlp.forward(xs),
        }
    }
}

struct LayerWeights {
    attention_wq: Arc<dyn QuantMethod>,
    attention_wk: Arc<dyn QuantMethod>,
    attention_wv: Arc<dyn QuantMethod>,
    attention_wo: Arc<dyn QuantMethod>,
    attention_norm: RmsNorm,
    mlp_or_moe: MlpOrMoe,
    ffn_norm: RmsNorm,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    rotary: Arc<RotaryEmbedding>,
    paged_attn: Option<PagedAttention>,
    sdpa_params: SdpaParams,
    dtype: DType,
}

impl LayerWeights {
    fn forward_attn(
        &self,
        x: &Tensor,
        mask: Option<&Tensor>,
        start_offsets: &[usize],
        kv_cache: &mut KvCache,
        metadata: Option<((Tensor, Tensor), &PagedAttentionInputMetadata)>,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, _) = x.dims3()?;

        // QKV projections with dtype conversion
        let q = MatMul
            .qmethod_matmul(x, &*self.attention_wq)?
            .to_dtype(self.dtype)?;
        let k = MatMul
            .qmethod_matmul(x, &*self.attention_wk)?
            .to_dtype(self.dtype)?;
        let v = MatMul
            .qmethod_matmul(x, &*self.attention_wv)?
            .to_dtype(self.dtype)?;

        // Reshape for multi-head attention
        let q = reshape_for_attn(q, b_sz, seq_len, self.n_head, self.head_dim)?;
        let k = reshape_for_attn(k, b_sz, seq_len, self.n_kv_head, self.head_dim)?;
        let v = reshape_for_attn(v, b_sz, seq_len, self.n_kv_head, self.head_dim)?;

        // RoPE
        let (q, k) = self.rotary.forward(&q, &k, start_offsets)?;

        // SDPA with cache
        let y = sdpa_with_cache(
            &q,
            &k,
            &v,
            mask,
            kv_cache,
            self.paged_attn.as_ref(),
            metadata,
            &self.sdpa_params,
        )?;

        // Reshape output and project
        let y = reshape_attn_output(y, b_sz, seq_len, mask.is_some())?;
        MatMul.qmethod_matmul(&y.to_dtype(x.dtype())?, &*self.attention_wo)
    }
}

#[deprecated(
    since = "0.8.0",
    note = "Use the unified Llama model instead. ModelWeights is a legacy GGUF-specific type."
)]
pub struct ModelWeights {
    tok_embeddings: Embedding,
    layers: Vec<LayerWeights>,
    /// Final norm layer (None for non-last pipeline stages)
    norm: Option<RmsNorm>,
    /// Output/LM head layer (None for non-last pipeline stages)
    output: Option<Arc<dyn QuantMethod>>,
    pub device: Device,
    pub max_seq_len: usize,
    mapper: Option<Box<dyn DeviceMapper + Send + Sync>>,
    dtype: DType,
}

impl ModelConfig::FromGGML for ModelWeights {
    fn from_ggml(mut ct: ggml_file::Content, gqa: usize, dtype: DType) -> Result<Self> {
        let head_dim = (ct.hparams.n_embd / ct.hparams.n_head) as usize;
        let rotary = RotaryEmbedding::new_partial(
            10000.,
            ct.hparams.n_rot as usize,
            DEFAULT_MAX_SEQ_LEN as usize,
            &ct.device,
            false,
            dtype,
        )?;
        let tok_embeddings = ct.remove("tok_embeddings.weight")?;
        let tok_embeddings = tok_embeddings.dequantize(&ct.device)?;
        let norm = RmsNorm::from_qtensor(ct.remove("norm.weight")?, 1e-5)?;
        let output = ct.remove("output.weight")?;
        let mut layers = Vec::with_capacity(ct.hparams.n_layer as usize);
        for layer_idx in NiceProgressBar::<_, 'b'>(
            0..ct.hparams.n_layer,
            "Loading repeating layers",
            &new_multi_progress(),
        ) {
            let prefix = format!("layers.{layer_idx}");
            let attention_wq = ct.remove(&format!("{prefix}.attention.wq.weight"))?;
            let attention_wk = ct.remove(&format!("{prefix}.attention.wk.weight"))?;
            let attention_wv = ct.remove(&format!("{prefix}.attention.wv.weight"))?;
            let attention_wo = ct.remove(&format!("{prefix}.attention.wo.weight"))?;
            let mlp_or_moe = {
                let feed_forward_w1 = ct.remove(&format!("{prefix}.feed_forward.w1.weight"))?;
                let feed_forward_w2 = ct.remove(&format!("{prefix}.feed_forward.w2.weight"))?;
                let feed_forward_w3 = ct.remove(&format!("{prefix}.feed_forward.w3.weight"))?;
                MlpOrMoe::Mlp(Mlp::from_weights(
                    Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                        q_weight: Arc::new(feed_forward_w1), // gate
                        b: None,
                    })?),
                    Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                        q_weight: Arc::new(feed_forward_w3), // up
                        b: None,
                    })?),
                    Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                        q_weight: Arc::new(feed_forward_w2), // down
                        b: None,
                    })?),
                    Activation::Silu,
                ))
            };
            let attention_norm = ct.remove(&format!("{prefix}.attention_norm.weight"))?;
            let ffn_norm = ct.remove(&format!("{prefix}.ffn_norm.weight"))?;
            let n_kv_head = ct.hparams.n_head as usize / gqa;
            layers.push(LayerWeights {
                attention_wq: Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                    q_weight: Arc::new(attention_wq),
                    b: None,
                })?),
                attention_wk: Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                    q_weight: Arc::new(attention_wk),
                    b: None,
                })?),
                attention_wv: Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                    q_weight: Arc::new(attention_wv),
                    b: None,
                })?),
                attention_wo: Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                    q_weight: Arc::new(attention_wo),
                    b: None,
                })?),
                attention_norm: RmsNorm::from_qtensor(attention_norm, 1e-5)?,
                mlp_or_moe,
                ffn_norm: RmsNorm::from_qtensor(ffn_norm, 1e-5)?,
                n_head: ct.hparams.n_head as usize,
                n_kv_head: ct.hparams.n_head as usize / gqa,
                head_dim: (ct.hparams.n_embd / ct.hparams.n_head) as usize,
                rotary: rotary.clone().into(),
                paged_attn: None, // TODO
                sdpa_params: SdpaParams {
                    n_kv_groups: ct.hparams.n_head as usize / n_kv_head,
                    softcap: None,
                    softmax_scale: 1.0 / (head_dim as f32).sqrt(),
                    sliding_window: None,
                },
                dtype,
            })
        }
        Ok(Self {
            tok_embeddings: Embedding::new(tok_embeddings, ct.hparams.n_embd as usize),
            layers,
            norm: Some(norm), // GGML doesn't support PP, always has norm
            output: Some(Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                q_weight: Arc::new(output),
                b: None,
            })?)),
            device: ct.device.clone(),
            max_seq_len: DEFAULT_MAX_SEQ_LEN as usize, // Cannot determine from ggml.
            mapper: None,
            dtype,
        })
    }
}

// llama `llm` fields:
// https://github.com/ggerganov/ggml/blob/master/docs/gguf.md#llm
// NOTE: Types here do not match spec
pub(crate) struct PropsGGUF {
    pub n_expert: usize,
    pub n_expert_used: usize,
    pub head_count: usize,
    pub head_count_kv: usize,
    pub block_count: usize,
    pub embedding_length: usize,
    pub feed_forward_length: usize,
    pub vocab_size: usize,
    pub rope_dim: usize,
    pub rms_norm_eps: f32,
    pub max_seq_len: usize,
    pub rope_freq_base: f32,
    pub key_length: usize,
    pub value_length: usize,
}

impl TryFrom<ContentMetadata<'_>> for PropsGGUF {
    type Error = anyhow::Error;

    fn try_from(c: ContentMetadata) -> std::result::Result<Self, Self::Error> {
        c.verify_arch("llama")?;

        let required = [
            "attention.head_count",
            "attention.head_count_kv",
            "block_count",
            "embedding_length",
            "rope.dimension_count",
            "attention.layer_norm_rms_epsilon",
        ];
        c.has_required_keys(&required)?;

        let embed_len = c.get_value::<u32>("embedding_length")? as usize;
        let head_count = c.get_value::<u32>("attention.head_count")? as usize;

        // NOTE: Values are not aligned with GGUFv3 types
        // TODO: Normalize value types to spec
        let props = Self {
            n_expert: c.get_value::<u32>("expert_count").ok().unwrap_or(0) as usize,
            n_expert_used: c.get_value::<u32>("expert_used_count").ok().unwrap_or(0) as usize,
            head_count,
            head_count_kv: c.get_value::<u32>("attention.head_count_kv")? as usize,
            block_count: c.get_value::<u32>("block_count")? as usize,
            embedding_length: embed_len,
            // feed_forward_length may not be present - default to 4x hidden (common ratio)
            feed_forward_length: c
                .get_value::<u32>("feed_forward_length")
                .ok()
                .map(|x| x as usize)
                .unwrap_or(embed_len * 4),
            // vocab_size: first try {arch}.vocab_size, then fall back to tokenizer tokens array length
            vocab_size: c
                .get_value::<u32>("vocab_size")
                .ok()
                .map(|x| x as usize)
                .or_else(|| {
                    c.metadata.get("tokenizer.ggml.tokens").and_then(|v| match v {
                        candle_core::quantized::gguf_file::Value::Array(arr) => Some(arr.len()),
                        _ => None,
                    })
                })
                .unwrap_or(32000), // Common default for Llama
            rope_dim: c.get_value::<u32>("rope.dimension_count")? as usize,
            // Strangely this value is generally 1e-6 in GGUF file but used to be 1e-5 by default.
            rms_norm_eps: c.get_value("attention.layer_norm_rms_epsilon")?,
            max_seq_len: c
                .get_value::<u64>("context_length")
                .ok()
                .unwrap_or(DEFAULT_MAX_SEQ_LEN as u64) as usize,
            rope_freq_base: c.get_value("rope.freq_base").ok().unwrap_or(10_000_f32),
            key_length: c
                .get_value::<u32>("attention.key_length")
                .ok()
                .map(|x| x as usize)
                .unwrap_or(embed_len / head_count),
            value_length: c
                .get_value::<u32>("attention.value_length")
                .ok()
                .map(|x| x as usize)
                .unwrap_or(embed_len / head_count),
        };

        Ok(props)
    }
}

use crate::models::LanguageModelConfig;

impl LanguageModelConfig for PropsGGUF {
    fn hidden_size(&self) -> usize {
        self.embedding_length
    }

    fn intermediate_size(&self) -> usize {
        self.feed_forward_length
    }

    fn num_layers(&self) -> usize {
        self.block_count
    }

    fn num_attention_heads(&self) -> usize {
        self.head_count
    }

    fn num_key_value_heads(&self) -> usize {
        self.head_count_kv
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    fn rms_norm_eps(&self) -> f64 {
        self.rms_norm_eps as f64
    }

    fn rope_theta(&self) -> f32 {
        self.rope_freq_base
    }

    fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    fn rope_dim(&self) -> usize {
        self.rope_dim
    }

    fn num_experts(&self) -> usize {
        self.n_expert
    }

    fn num_experts_used(&self) -> usize {
        self.n_expert_used
    }
}

impl ModelConfig::FromGGUF for ModelWeights {
    fn from_gguf<R: std::io::Seek + std::io::Read>(
        mut ct: Content<'_, R>,
        device: &Device,
        mapper: Box<dyn DeviceMapper + Send + Sync>,
        attention_mechanism: AttentionImplementation,
        dtype: DType,
        layer_range: Option<std::ops::Range<usize>>,
        _adapter_registry: Option<std::sync::Arc<crate::lora::AdapterRegistry>>,
    ) -> Result<Self> {
        // Parameter extraction from metadata.
        let metadata = ContentMetadata {
            path_prefix: "llama",
            metadata: ct.get_metadata(),
        };
        let PropsGGUF {
            n_expert,
            n_expert_used,
            head_count,
            head_count_kv,
            block_count,
            embedding_length,
            feed_forward_length: _,  // Used via LlamaConfig trait
            vocab_size: _,           // Used via LlamaConfig trait
            rope_dim,
            rms_norm_eps,
            max_seq_len,
            rope_freq_base,
            key_length,
            value_length,
        } = PropsGGUF::try_from(metadata).or_else(|err| candle_core::bail!("{err}"))?;

        // Determine layer range for partial loading (pipeline parallelism)
        let total_layers = block_count;
        let layer_range = layer_range.unwrap_or(0..total_layers);
        let layer_start = layer_range.start;
        let layer_end = layer_range.end.min(total_layers);
        let num_loaded_layers = layer_end - layer_start;

        if layer_start > 0 || layer_end < total_layers {
            tracing::info!(
                "Pipeline parallelism: loading layers {}..{} of {} total",
                layer_start,
                layer_end,
                total_layers
            );
        }

        let qtok_embeddings = ct.tensor("token_embd.weight", device)?;
        let tok_embeddings = qtok_embeddings.dequantize(device)?;

        // PP: Only load norm and output (LM head) for last stage
        let is_last_stage = layer_end >= total_layers;
        let norm = if is_last_stage {
            Some(RmsNorm::from_qtensor(ct.tensor("output_norm.weight", device)?, rms_norm_eps)?)
        } else {
            None
        };
        let output = if is_last_stage {
            Some(if !ct.has_tensor("output.weight") {
                ct.tensor("token_embd.weight", device)?
            } else {
                ct.tensor("output.weight", device)?
            })
        } else {
            None
        };

        let mut layers = Vec::with_capacity(num_loaded_layers);

        let head_dim = key_length;
        if key_length != value_length {
            candle_core::bail!(
                "Expected key_length == value_length, got {key_length} != {value_length}"
            );
        }

        // Only create RoPE embeddings for loaded layers
        let mut ropes = HashMap::new();
        for layer_idx in layer_start..layer_end {
            let device = mapper.device_for(layer_idx, false).unwrap_or(device);
            ropes.insert(
                device.location(),
                Arc::new(RotaryEmbedding::new(
                    rope_freq_base,
                    rope_dim,
                    max_seq_len,
                    device,
                    false,
                    dtype,
                )?),
            );
        }

        // Only load layers in the specified range
        for layer_idx in NiceProgressBar::<_, 'b'>(
            layer_start..layer_end,
            "Loading repeating layers",
            &new_multi_progress(),
        ) {
            let prefix = format!("blk.{layer_idx}");
            let device = mapper.device_for(layer_idx, false).unwrap_or(device);
            let rotary = ropes
                .get(&device.location())
                .expect("No RoPE for device location!")
                .clone();

            let attention_wq = ct.tensor(&format!("{prefix}.attn_q.weight"), device)?;
            let attention_wk = ct.tensor(&format!("{prefix}.attn_k.weight"), device)?;
            let attention_wv = ct.tensor(&format!("{prefix}.attn_v.weight"), device)?;
            let attention_wo = ct.tensor(&format!("{prefix}.attn_output.weight"), device)?;
            let mlp_or_moe = if n_expert <= 1 {
                let ffn_gate = ct.tensor(&format!("{prefix}.ffn_gate.weight"), device)?;
                let ffn_down = ct.tensor(&format!("{prefix}.ffn_down.weight"), device)?;
                let ffn_up = ct.tensor(&format!("{prefix}.ffn_up.weight"), device)?;
                MlpOrMoe::Mlp(Mlp::from_weights(
                    Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                        q_weight: Arc::new(ffn_gate),
                        b: None,
                    })?),
                    Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                        q_weight: Arc::new(ffn_up),
                        b: None,
                    })?),
                    Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                        q_weight: Arc::new(ffn_down),
                        b: None,
                    })?),
                    Activation::Silu,
                ))
            } else {
                let feed_forward_gate_inp =
                    ct.tensor(&format!("{prefix}.ffn_gate_inp.weight"), device)?;
                let mut experts = Vec::with_capacity(n_expert);
                match ct.tensor(&format!("{prefix}.ffn_gate_exps.weight"), device) {
                    Ok(feed_forward_gate_exps) => {
                        let feed_forward_down_exps =
                            ct.tensor(&format!("{prefix}.ffn_down_exps.weight"), device)?;
                        let feed_forward_up_exps =
                            ct.tensor(&format!("{prefix}.ffn_up_exps.weight"), device)?;

                        let dequant_ffn_gate = feed_forward_gate_exps
                            .dequantize(device)?
                            .chunk(n_expert, 0)?;
                        let dequant_ffn_down = feed_forward_down_exps
                            .dequantize(device)?
                            .chunk(n_expert, 0)?;
                        let dequant_ffn_up = feed_forward_up_exps
                            .dequantize(device)?
                            .chunk(n_expert, 0)?;

                        assert_eq!(dequant_ffn_up.len(), dequant_ffn_down.len());
                        assert_eq!(dequant_ffn_gate.len(), dequant_ffn_down.len());
                        assert_eq!(dequant_ffn_gate.len(), n_expert);

                        let gate_type = feed_forward_gate_exps.dtype();
                        let down_type = feed_forward_down_exps.dtype();
                        let up_type = feed_forward_up_exps.dtype();

                        for (ff_gate, (ff_down, ff_up)) in dequant_ffn_gate
                            .into_iter()
                            .zip(dequant_ffn_down.into_iter().zip(dequant_ffn_up))
                        {
                            experts.push(Mlp::from_weights(
                                Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                                    q_weight: Arc::new(QTensor::quantize(&ff_gate, gate_type)?),
                                    b: None,
                                })?),
                                Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                                    q_weight: Arc::new(QTensor::quantize(&ff_up, up_type)?),
                                    b: None,
                                })?),
                                Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                                    q_weight: Arc::new(QTensor::quantize(&ff_down, down_type)?),
                                    b: None,
                                })?),
                                Activation::Silu,
                            ))
                        }
                    }
                    Err(_) => {
                        for i in 0..n_expert {
                            let ffn_gate =
                                ct.tensor(&format!("{prefix}.ffn_gate.{i}.weight"), device)?;
                            let ffn_down =
                                ct.tensor(&format!("{prefix}.ffn_down.{i}.weight"), device)?;
                            let ffn_up =
                                ct.tensor(&format!("{prefix}.ffn_up.{i}.weight"), device)?;
                            experts.push(Mlp::from_weights(
                                Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                                    q_weight: Arc::new(ffn_gate),
                                    b: None,
                                })?),
                                Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                                    q_weight: Arc::new(ffn_up),
                                    b: None,
                                })?),
                                Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                                    q_weight: Arc::new(ffn_down),
                                    b: None,
                                })?),
                                Activation::Silu,
                            ))
                        }
                    }
                }
                MlpOrMoe::MoE {
                    n_expert_used,
                    feed_forward_gate_inp: Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                        q_weight: Arc::new(feed_forward_gate_inp),
                        b: None,
                    })?),
                    experts,
                }
            };
            let attention_norm = ct.tensor(&format!("{prefix}.attn_norm.weight"), device)?;
            let ffn_norm = ct.tensor(&format!("{prefix}.ffn_norm.weight"), device)?;
            let paged_attn = match &attention_mechanism {
                AttentionImplementation::Eager => None,
                AttentionImplementation::PagedAttention => {
                    Some(PagedAttention::new(head_dim, device, None)?)
                }
            };
            layers.push(LayerWeights {
                attention_wq: Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                    q_weight: Arc::new(attention_wq),
                    b: None,
                })?),
                attention_wk: Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                    q_weight: Arc::new(attention_wk),
                    b: None,
                })?),
                attention_wv: Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                    q_weight: Arc::new(attention_wv),
                    b: None,
                })?),
                attention_wo: Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                    q_weight: Arc::new(attention_wo),
                    b: None,
                })?),
                attention_norm: RmsNorm::from_qtensor(attention_norm, rms_norm_eps)?,
                mlp_or_moe,
                ffn_norm: RmsNorm::from_qtensor(ffn_norm, rms_norm_eps)?,
                n_head: head_count,
                n_kv_head: head_count_kv,
                head_dim,
                rotary: rotary.clone(),
                paged_attn,
                sdpa_params: SdpaParams {
                    n_kv_groups: head_count / head_count_kv,
                    softcap: None,
                    softmax_scale: 1.0 / (head_dim as f32).sqrt(),
                    sliding_window: None,
                },
                dtype,
            })
        }
        Ok(Self {
            tok_embeddings: Embedding::new(tok_embeddings, embedding_length),
            layers,
            norm,
            output: output.map(|q_tensor| {
                Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                    q_weight: Arc::new(q_tensor),
                    b: None,
                }).unwrap()) as Arc<dyn QuantMethod>
            }),
            device: device.clone(),
            max_seq_len,
            mapper: Some(mapper),
            dtype,
        })
    }
}

impl ModelWeights {
    /// Run transformer layers on hidden states.
    ///
    /// This is the core layer iteration logic used by `transform()`.
    fn run_layers(
        &self,
        mut hidden: Tensor,
        mask: Option<&Tensor>,
        position_offsets: &[usize],
        metadata: Option<(&[(Tensor, Tensor)], &PagedAttentionInputMetadata)>,
        cache: &mut [KvCache],
    ) -> Result<Tensor> {
        for (i, layer) in self.layers.iter().enumerate() {
            if let Some(ref mapper) = self.mapper {
                hidden = mapper.map(hidden, i)?;
            }
            let residual = &hidden;
            let x = layer.attention_norm.forward(&hidden)?;
            let attn = layer.forward_attn(
                &x,
                mask.map(|m| m.to_device(x.device()).unwrap())
                    .as_ref(),
                position_offsets,
                &mut cache[i],
                metadata
                    .as_ref()
                    .map(|(kv_cache, meta)| (kv_cache[i].clone(), *meta)),
            )?;
            let x = (attn + residual)?;
            let residual = &x;
            let x = layer.ffn_norm.forward(&x)?;
            let x = layer.mlp_or_moe.forward(&x)?;
            hidden = (x + residual)?;
        }

        Ok(hidden)
    }
}

// ============================================================================
// Model Trait Implementations
// ============================================================================

impl Model for ModelWeights {
    fn device(&self) -> &Device {
        &self.device
    }
}

impl TransformerModel for ModelWeights {
    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    fn embed(&self, tokens: &Tensor) -> Result<Tensor> {
        self.tok_embeddings.forward(tokens)
    }

    fn transform(&self, hidden: Tensor, ctx: &TransformContext, cache: &mut [KvCache]) -> Result<Tensor> {
        let seq_len = hidden.dim(1)?;
        let start_offsets: Vec<usize> = vec![ctx.position_offset];

        // Compute causal mask using position offsets
        let mask = CausalMasker.make_causal_mask_as(
            seq_len,
            hidden.device(),
            &start_offsets.as_slice() as &dyn PastKvLenCache,
            self.dtype,
        )?;
        // Only apply mask on first prompt chunk (optimization for paged attention)
        let mask = mask.filter(|_| {
            ctx.paged_attn
                .as_ref()
                .map(|pa| pa.metadata.is_first_prompt_chunk)
                .unwrap_or(true)
        });

        // Run transformer layers
        let meta_ref = ctx
            .paged_attn
            .as_ref()
            .map(|pa| (pa.kv_cache.as_slice(), pa.metadata));
        self.run_layers(hidden, mask.as_ref(), &start_offsets, meta_ref, cache)
    }
}

impl LanguageModel for ModelWeights {
    fn lm_head(&self, hidden: Tensor) -> Result<Tensor> {
        // Move to model device and apply final norm
        let hidden = hidden.to_device(&self.device)?;
        let x = self
            .norm
            .as_ref()
            .expect("lm_head called on non-last pipeline stage")
            .forward(&hidden)?;
        // Project to vocabulary
        MatMul.qmethod_matmul(
            &x.contiguous()?,
            &**self
                .output
                .as_ref()
                .expect("lm_head called on non-last pipeline stage"),
        )
    }
}
