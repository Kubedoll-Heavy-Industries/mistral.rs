#![allow(clippy::cast_possible_truncation)]

use std::{any::Any, sync::Arc};

use anyhow::Result;
use candle_core::Device;
use text_models_inputs_processor::PagedAttentionMeta;
use tokenizers::Tokenizer;

use crate::{device_map::DeviceMapper, sequence::Sequence};

#[derive(PartialEq)]
pub enum InputsProcessorType {
    Text,
    Vision,
    Embedding,
}

pub struct InputProcessorOutput {
    pub inputs: Box<dyn Any>,
    pub seq_indices: Vec<usize>,
}

/// Processor: Prepare inputs for the model (potentially preparing the images if applicable)
pub trait InputsProcessor {
    /// This should also enable matmul via f16 if prompt and the sequence length is greater than 32.
    /// Otherwise, matmul via f16 is disabled.
    ///
    /// This should return a type which can be downcasted to the proper type as used in `forward_inputs`
    #[allow(clippy::too_many_arguments)]
    fn process_inputs(
        &self,
        tokenizer: Option<Arc<Tokenizer>>,
        input_seqs: &mut [&mut Sequence],
        is_prompt: bool,
        is_xlora: bool,
        device: &Device,
        no_kv_cache: bool,
        last_n_context_len: Option<(usize, usize)>,
        return_raw_logits: bool,
        other_config: Option<Arc<dyn Any>>,
        paged_attn_metadata: Option<PagedAttentionMeta>,
        mapper: Option<&dyn DeviceMapper>,
    ) -> Result<InputProcessorOutput>;

    fn get_type(&self) -> InputsProcessorType;
}

// ========================= Test models input processor

pub mod text_models_inputs_processor {
    use std::{any::Any, collections::HashMap, fmt::Debug, sync::Arc};

    use anyhow::Result;
    use candle_core::{DType, Device, DeviceLocation, Tensor, WithDType};
    use tokenizers::Tokenizer;

    use crate::{
        device_map::DeviceMapper,
        get_mut_arcmutex,
        paged_attention::{BlockEngine, _PAD_SLOT_ID},
        sequence::Sequence,
    };

    use super::{InputProcessorOutput, InputsProcessor, InputsProcessorType};

    fn _make_tensor_with_pad<D: WithDType>(
        x: Vec<Vec<D>>,
        max_len: usize,
        pad: D,
        device: &Device,
    ) -> Result<Tensor> {
        let mut padded_x = Vec::new();
        for mut x_i in x {
            assert!(x_i.len() <= max_len);
            x_i.extend([pad].repeat(max_len - x_i.len()));
            let shape = (x_i.len(),);
            padded_x.push(Tensor::from_vec(x_i, shape, device)?);
        }
        Tensor::cat(&padded_x[..], 0).map_err(anyhow::Error::msg)
    }

    pub struct PagedAttentionMeta {
        pub sliding_window: Option<usize>,
        pub block_size: usize,
        pub block_engine: Arc<tokio::sync::Mutex<BlockEngine>>,
    }

    #[derive(Clone, Debug)]
    #[allow(dead_code)]
    pub struct PagedAttentionInputMetadata {
        pub block_tables: Option<HashMap<DeviceLocation, Tensor>>,
        pub context_lens: Option<HashMap<DeviceLocation, Tensor>>,
        pub slot_mappings: HashMap<DeviceLocation, Tensor>,
        pub max_context_len: Option<usize>,
        pub is_first_prompt_chunk: bool,
    }

    impl PagedAttentionInputMetadata {
        /// Create a dummy input metadata, assuming that this will NOT be used for decoding.
        /// This is used for the case of imatrix generation.
        pub fn dummy(dev: &Device) -> candle_core::Result<Self> {
            Ok(PagedAttentionInputMetadata {
                block_tables: None,
                context_lens: None,
                max_context_len: None,
                slot_mappings: HashMap::from([(dev.location(), Tensor::new(&[0f32], dev)?)]),
                is_first_prompt_chunk: true,
            })
        }
    }

    #[derive(Clone, Debug)]
    pub struct FlashParams {
        pub max_q: u32,
        pub max_k: u32,
        pub cumulative_seqlens_q: HashMap<DeviceLocation, Tensor>,
        pub cumulative_seqlens_k: HashMap<DeviceLocation, Tensor>,
        pub causal: bool,
    }

    pub struct InputMetadata {
        pub input: Tensor,
        pub positions: Vec<usize>,
        pub context_lens: Vec<(usize, usize)>, // (start index, len)
        pub position_ids: Vec<usize>,
        pub paged_attn_meta: Option<PagedAttentionInputMetadata>, // For paged attention
        pub flash_meta: FlashParams,
    }

    pub struct InnerInputProcessorOutput {
        pub inputs: InputMetadata,
        pub seq_indices: Vec<usize>,
    }

    // chunk_offset_toks is the number of tokens by which the tokens are offset,
    // chunk_offset_toks / prompt_chunksize = number of batches
    #[allow(clippy::too_many_arguments)]
    pub fn make_prompt_chunk<T: WithDType + Debug>(
        chunk_offset_toks: usize,
        toks: Vec<&[T]>,
        seq_ids: &[usize],
        device: &Device,
        last_n_context_len: Option<(usize, usize)>,
        return_raw_logits: bool,
        mut paged_attn_metadata: Option<&mut PagedAttentionMeta>,
        mapper: Option<&dyn DeviceMapper>,
    ) -> Result<InputMetadata> {
        let max_len = toks
            .iter()
            .map(|seq| seq.len())
            .max()
            .expect("No sequences");
        let padding_tok = T::zero();
        // Pad each sequence by the padding token to the max len.
        let mut seqs_tensors = Vec::new();
        let mut seqlen_offsets = Vec::new();
        let mut context_lens = Vec::new();
        let mut position_ids = Vec::new();
        let mut slot_mappings = Vec::new();
        let mut block_tables = Vec::new();
        let mut paged_attn_context_lens = Vec::new();
        let flash_attn = crate::using_flash_attn();
        let mut seqlens_q = if flash_attn { vec![0] } else { Vec::new() };
        let mut seqlens_k = if flash_attn { vec![0] } else { Vec::new() };
        for (seq_id, ctxt) in seq_ids.iter().zip(toks) {
            let prompt_len = ctxt.len();
            let offset = last_n_context_len.unwrap_or_default();
            seqlen_offsets.push(offset.1 + chunk_offset_toks);

            position_ids.push(ctxt.len() + chunk_offset_toks);
            let mut ctxt = ctxt.to_vec();
            ctxt.extend(std::iter::repeat_n(
                padding_tok,
                max_len.saturating_sub(ctxt.len()),
            ));
            // If we are returning raw logits, we want to not trim the logits at all.
            if return_raw_logits {
                if last_n_context_len.is_some() {
                    anyhow::bail!("`return_raw_logits` is incompatible with `last_n_context_len`");
                }

                context_lens.push((0, ctxt.len()));
            } else {
                context_lens.push((
                    ctxt.len()
                        .saturating_sub(last_n_context_len.map(|(a, _)| a).unwrap_or(1)),
                    last_n_context_len.map(|(a, _)| a).unwrap_or(1),
                ));
            }

            if flash_attn {
                seqlens_q.push(ctxt.len() as u32);
                seqlens_k.push((ctxt.len() + chunk_offset_toks) as u32);
            }

            seqs_tensors.push(Tensor::new(ctxt, device)?.unsqueeze(0)?);

            if let Some(paged_attn_metadata) = &mut paged_attn_metadata {
                let block_engine = get_mut_arcmutex!(paged_attn_metadata.block_engine);
                let table = block_engine.block_tables.get(seq_id);

                if table.is_none() {
                    // Will be None during profiling.
                    slot_mappings.push([_PAD_SLOT_ID].repeat(prompt_len));
                    continue;
                }
                let table = table
                    .unwrap()
                    .iter()
                    .map(|block| block.block_id())
                    .collect::<Vec<_>>();

                let start_idx = if let Some(sliding_window) = paged_attn_metadata.sliding_window {
                    prompt_len.saturating_sub(sliding_window)
                } else {
                    0
                };

                let mut slot_mapping = Vec::new();
                let mut ctxt_len = Vec::new();
                for i in chunk_offset_toks..prompt_len + chunk_offset_toks {
                    if i < start_idx {
                        // Pad [0,start_idx) with _PAD_TOKEN_ID
                        slot_mapping.push(_PAD_SLOT_ID);
                    }
                    ctxt_len.push(i);

                    let block_number = if i / paged_attn_metadata.block_size >= table.len() {
                        panic!(
                            "Block table is too small (prompt)! i={} block_size={} table_len={}",
                            i,
                            paged_attn_metadata.block_size,
                            table.len()
                        );
                    } else {
                        table.get(i / paged_attn_metadata.block_size).unwrap()
                    };
                    let block_offset = i % paged_attn_metadata.block_size;
                    // Use checked arithmetic to prevent overflow
                    let slot = block_number
                        .checked_mul(paged_attn_metadata.block_size)
                        .and_then(|v| v.checked_add(block_offset))
                        .expect("Slot calculation overflowed");
                    slot_mapping.push(
                        slot.try_into()
                            .expect("Slot value too large for target integer type"),
                    );
                    block_tables.push(table.clone());
                }
                slot_mappings.push(slot_mapping);
                paged_attn_context_lens.push(ctxt_len);
            }
        }

        let (max_q, max_k, seqlens_q_map, seqlens_k_map) = if flash_attn {
            // SAFETY: seqlens_q/k are initialized with vec![0] when flash_attn is true,
            // so they are guaranteed to be non-empty here.
            let max_q = *seqlens_q
                .iter()
                .max()
                .expect("seqlens_q should not be empty when flash_attn is enabled");
            let max_k = *seqlens_k
                .iter()
                .max()
                .expect("seqlens_k should not be empty when flash_attn is enabled");
            // Create tensors on CPU first to avoid CUDA context issues when copying
            // between different GPU devices. Each GPU has its own CUDA context, and
            // candle/cudarc doesn't properly switch contexts when doing GPU-to-GPU
            // transfers (which go through CPU). By creating on CPU first, we avoid
            // the cross-context memory access that causes CUDA_ERROR_INVALID_VALUE.
            let seqlens_q = Tensor::new(seqlens_q, &Device::Cpu)?
                .to_dtype(DType::F32)?
                .cumsum(0)?
                .to_dtype(DType::U32)?;
            let seqlens_k = Tensor::new(seqlens_k, &Device::Cpu)?
                .to_dtype(DType::F32)?
                .cumsum(0)?
                .to_dtype(DType::U32)?;

            let mut seqlens_q_map = HashMap::new();
            let mut seqlens_k_map = HashMap::new();

            let devices = mapper.unwrap().get_unique_devices();
            for device in devices {
                seqlens_q_map.insert(device.location(), seqlens_q.to_device(&device)?);
                seqlens_k_map.insert(device.location(), seqlens_k.to_device(&device)?);
            }
            (max_q, max_k, seqlens_q_map, seqlens_k_map)
        } else {
            (0, 0, HashMap::new(), HashMap::new())
        };

        let input = Tensor::cat(&seqs_tensors[..], 0)?;

        let paged_attn_meta = if paged_attn_metadata.is_some() {
            // Create paged attention tensors on CPU first (see comment above about CUDA contexts)
            let max_slot_mapping_len = slot_mappings.iter().map(|x| x.len()).max().unwrap();
            let slot_mappings = _make_tensor_with_pad(
                slot_mappings,
                max_slot_mapping_len,
                _PAD_SLOT_ID,
                &Device::Cpu,
            )?;

            let max_block_table_len = block_tables.iter().map(|x| x.len()).max().unwrap();
            let block_tables = _make_tensor_with_pad(
                block_tables
                    .iter()
                    .map(|x| x.iter().map(|x| *x as u32).collect::<Vec<_>>())
                    .collect::<Vec<_>>(),
                max_block_table_len,
                0,
                &Device::Cpu,
            )?;
            let block_tables = block_tables.reshape(((), max_block_table_len))?;

            let max_context_len = paged_attn_context_lens
                .iter()
                .map(|x| x.len())
                .max()
                .unwrap();

            let context_lens = _make_tensor_with_pad(
                paged_attn_context_lens
                    .iter()
                    .map(|x| x.iter().map(|x| *x as u32).collect::<Vec<_>>())
                    .collect::<Vec<_>>(),
                max_context_len,
                0,
                &Device::Cpu,
            )?
            .reshape(((),))?;

            // For device mapping, make a copy of each tensor for each device
            let devices = mapper.unwrap().get_unique_devices();
            let mut slot_mappings_map = HashMap::new();
            let mut block_tables_map = HashMap::new();
            let mut context_lens_map = HashMap::new();

            for device in devices {
                slot_mappings_map
                    .insert(device.location(), slot_mappings.clone().to_device(&device)?);
                block_tables_map
                    .insert(device.location(), block_tables.clone().to_device(&device)?);
                context_lens_map
                    .insert(device.location(), context_lens.clone().to_device(&device)?);
            }

            Some(PagedAttentionInputMetadata {
                slot_mappings: slot_mappings_map,
                block_tables: Some(block_tables_map),
                context_lens: Some(context_lens_map),
                max_context_len: Some(max_context_len),
                is_first_prompt_chunk: chunk_offset_toks == 0,
            })
        } else {
            None
        };

        Ok(InputMetadata {
            input,
            positions: seqlen_offsets,
            context_lens,
            position_ids,
            paged_attn_meta,
            flash_meta: FlashParams {
                max_k,
                max_q,
                cumulative_seqlens_k: seqlens_k_map,
                cumulative_seqlens_q: seqlens_q_map,
                causal: true,
            },
        })
    }

    fn make_completion_chunk<T: WithDType>(
        toks: Vec<&[T]>,
        input_seqs: &[&mut Sequence],
        device: &Device,
        mut paged_attn_metadata: Option<&mut PagedAttentionMeta>,
        mapper: Option<&dyn DeviceMapper>,
    ) -> Result<InputMetadata> {
        // Pad each sequence by the padding token to the max len.
        let flash_attn = crate::using_flash_attn();
        let mut seqs_tensors = Vec::new();
        let mut seqlen_offsets = Vec::new();
        let mut context_lens = Vec::new();
        let mut position_ids = Vec::new();

        let mut slot_mappings = Vec::new();
        let mut block_tables = Vec::new();
        let mut paged_attn_context_lens = Vec::new();
        let mut seqlens_q = if flash_attn { vec![0] } else { Vec::new() };
        let mut seqlens_k = if flash_attn { vec![0] } else { Vec::new() };
        for (seq, ctxt) in input_seqs.iter().zip(toks) {
            // For decode: position = max(token_offset, seq.len() - 1)
            // - Normal inference: token_offset=0, seq.len()=full, so max(0, n-1) = n-1 ✓
            // - Prefix cache: token_offset=cached, seq.len()=full, so max(3, n-1) = n-1 ✓
            // - PP Stage 1: token_offset=position_from_stage0, seq.len()=small, so max(223, 63) = 223 ✓
            let start_pos = std::cmp::max(seq.token_offset(), seq.len().saturating_sub(1));
            let ctxt_offset = ctxt.len().saturating_sub(1);
            let ctxt = ctxt[ctxt_offset..].to_vec();
            seqlen_offsets.push(start_pos);
            context_lens.push((0, 1));
            position_ids.push(seq.len());

            if flash_attn {
                seqlens_q.push(ctxt.len() as u32);
                seqlens_k.push((ctxt.len() + start_pos) as u32);
            }

            seqs_tensors.push(Tensor::new(ctxt, device)?.unsqueeze(0)?);

            if let Some(paged_attn_metadata) = &mut paged_attn_metadata {
                let block_engine = get_mut_arcmutex!(paged_attn_metadata.block_engine);
                let table = block_engine.block_tables.get(seq.id()).unwrap();

                let table = table
                    .iter()
                    .map(|block| block.block_id())
                    .collect::<Vec<_>>();

                let block_pos = start_pos - seq.token_offset();
                let block_number = if block_pos / paged_attn_metadata.block_size >= table.len() {
                    panic!("Block table is too small (completion)! start_pos={} block_size={} table_len={}", block_pos, paged_attn_metadata.block_size, table.len());
                } else {
                    table
                        .get(block_pos / paged_attn_metadata.block_size)
                        .unwrap()
                };
                let block_offset = block_pos % paged_attn_metadata.block_size;
                // Use checked arithmetic to prevent overflow
                let slot = block_number
                    .checked_mul(paged_attn_metadata.block_size)
                    .and_then(|v| v.checked_add(block_offset))
                    .expect("Slot calculation overflowed");
                let slot = slot
                    .try_into()
                    .expect("Slot value too large for target integer type");
                slot_mappings.push(vec![slot]);

                if let Some(sliding_window) = paged_attn_metadata.sliding_window {
                    let sliding_window_blocks = sliding_window / paged_attn_metadata.block_size;
                    let slide_idx = if table.len() > sliding_window_blocks {
                        table.len() - sliding_window_blocks
                    } else {
                        0
                    };
                    block_tables.push(table.get(slide_idx..).unwrap().to_vec());
                } else {
                    block_tables.push(table);
                }

                // Total context = token_offset (prefix/PP position) + current buffer length
                // This correctly handles: normal (0+N), prefix cache (cached+rest), PP (pos+1)
                let total_context = seq.token_offset() + seq.len();
                let paged_attn_context_len =
                    if let Some(sliding_window) = paged_attn_metadata.sliding_window {
                        total_context.min(sliding_window)
                    } else {
                        total_context
                    };
                paged_attn_context_lens.push(paged_attn_context_len);
            }
        }

        let (max_q, max_k, seqlens_q_map, seqlens_k_map) = if flash_attn {
            // SAFETY: seqlens_q/k are initialized with vec![0] when flash_attn is true,
            // so they are guaranteed to be non-empty here.
            let max_q = *seqlens_q
                .iter()
                .max()
                .expect("seqlens_q should not be empty when flash_attn is enabled");
            let max_k = *seqlens_k
                .iter()
                .max()
                .expect("seqlens_k should not be empty when flash_attn is enabled");
            // Create tensors on CPU first to avoid CUDA context issues (see make_prompt_chunk)
            let seqlens_q = Tensor::new(seqlens_q, &Device::Cpu)?
                .to_dtype(DType::F32)?
                .cumsum(0)?
                .to_dtype(DType::U32)?;
            let seqlens_k = Tensor::new(seqlens_k, &Device::Cpu)?
                .to_dtype(DType::F32)?
                .cumsum(0)?
                .to_dtype(DType::U32)?;

            let mut seqlens_q_map = HashMap::new();
            let mut seqlens_k_map = HashMap::new();

            let devices = mapper.unwrap().get_unique_devices();
            for device in devices {
                seqlens_q_map.insert(device.location(), seqlens_q.to_device(&device)?);
                seqlens_k_map.insert(device.location(), seqlens_k.to_device(&device)?);
            }
            (max_q, max_k, seqlens_q_map, seqlens_k_map)
        } else {
            (0, 0, HashMap::new(), HashMap::new())
        };

        let paged_attn_meta = if paged_attn_metadata.is_some() {
            // Create paged attention tensors on CPU first (see make_prompt_chunk for explanation)
            let slot_mappings =
                _make_tensor_with_pad(slot_mappings, 1, _PAD_SLOT_ID, &Device::Cpu)?;

            let max_block_table_len = block_tables
                .iter()
                .map(|x| x.len())
                .max()
                .expect("block_tables should not be empty when paged attention is enabled");

            let block_tables = _make_tensor_with_pad(
                block_tables
                    .iter()
                    .map(|x| x.iter().map(|x| *x as u32).collect::<Vec<_>>())
                    .collect::<Vec<_>>(),
                max_block_table_len,
                0,
                &Device::Cpu,
            )?;
            let block_tables = block_tables.reshape(((), max_block_table_len))?;

            let max_context_len = paged_attn_context_lens.iter().max().unwrap();

            let context_lens = Tensor::from_vec(
                paged_attn_context_lens
                    .iter()
                    .map(|x| *x as u32)
                    .collect::<Vec<_>>(),
                (paged_attn_context_lens.len(),),
                &Device::Cpu,
            )?;

            // For device mapping, make a copy of each tensor for each device
            let devices = mapper.unwrap().get_unique_devices();
            let mut slot_mappings_map = HashMap::new();
            let mut block_tables_map = HashMap::new();
            let mut context_lens_map = HashMap::new();

            for device in devices {
                slot_mappings_map
                    .insert(device.location(), slot_mappings.clone().to_device(&device)?);
                block_tables_map
                    .insert(device.location(), block_tables.clone().to_device(&device)?);
                context_lens_map
                    .insert(device.location(), context_lens.clone().to_device(&device)?);
            }

            Some(PagedAttentionInputMetadata {
                slot_mappings: slot_mappings_map,
                block_tables: Some(block_tables_map),
                context_lens: Some(context_lens_map),
                max_context_len: Some(*max_context_len),
                is_first_prompt_chunk: false,
            })
        } else {
            None
        };

        Ok(InputMetadata {
            input: Tensor::cat(&seqs_tensors, 0)?,
            positions: seqlen_offsets,
            context_lens,
            position_ids,
            paged_attn_meta,
            flash_meta: FlashParams {
                max_k,
                max_q,
                cumulative_seqlens_k: seqlens_k_map,
                cumulative_seqlens_q: seqlens_q_map,
                causal: true,
            },
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn get_prompt_input<T: WithDType + std::fmt::Debug>(
        toks: Vec<&[T]>,
        input_seqs: &[&mut Sequence],
        device: &Device,
        last_n_context_len: Option<(usize, usize)>,
        return_raw_logits: bool,
        paged_attn_metadata: Option<&mut PagedAttentionMeta>,
        mapper: Option<&dyn DeviceMapper>,
    ) -> Result<InnerInputProcessorOutput> {
        // KV/RoPE position offset calculation:
        // - Normal + prefix cache: token_offset is the cached prefix length.
        // - Chunked prefill: prefill_chunk_offset is the start of the current chunk.
        // - Pipeline continuation stages: token_offset is set to the absolute position coming
        //   from stage 0, and prefill_chunk_offset remains 0.
        //
        // The invariant we want is: offset == absolute KV position for the first token in `toks`.
        let offset = input_seqs[0].token_offset() + input_seqs[0].prefill_chunk_offset();

        make_prompt_chunk(
            offset,
            toks,
            &input_seqs.iter().map(|s| *s.id()).collect::<Vec<_>>(),
            device,
            last_n_context_len,
            return_raw_logits,
            paged_attn_metadata,
            mapper,
        )
        .map(|inputs| InnerInputProcessorOutput {
            inputs,
            seq_indices: (0..input_seqs.len()).collect(),
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn get_completion_input<T: WithDType + std::fmt::Debug>(
        toks: Vec<&[T]>,
        input_seqs: &[&mut Sequence],
        device: &Device,
        no_kv_cache: bool,
        last_n_context_len: Option<(usize, usize)>,
        return_raw_logits: bool,
        paged_attn_metadata: Option<&mut PagedAttentionMeta>,
        mapper: Option<&dyn DeviceMapper>,
    ) -> Result<InnerInputProcessorOutput> {
        if no_kv_cache {
            return get_prompt_input(
                toks,
                input_seqs,
                device,
                last_n_context_len,
                return_raw_logits,
                paged_attn_metadata,
                mapper,
            );
        }

        make_completion_chunk(toks, input_seqs, device, paged_attn_metadata, mapper).map(|inputs| {
            InnerInputProcessorOutput {
                inputs,
                seq_indices: (0..input_seqs.len()).collect(),
            }
        })
    }


    /// State machine for inference: prefill (with chunking) or decode.
    /// Carries structured metadata instead of booleans and guessing from tensor shapes.
    #[derive(Clone, Debug)]
    pub enum InferenceStep {
        /// Processing prompt tokens (possibly in chunks for long prompts).
        Prefill {
            /// Tokens in current chunk
            chunk_tokens: Vec<u32>,
            /// Cumulative position where this chunk starts (KV cache position)
            chunk_start_position: usize,
            /// Total tokens in the original prompt
            total_prompt_tokens: usize,
        },
        /// Generating output tokens one at a time.
        Decode {
            /// The single token being processed
            token: u32,
            /// Current KV cache position
            position: usize,
        },
    }

    impl InferenceStep {
        /// Check if this is the final chunk of prefill.
        /// For decode steps, always returns true.
        pub fn is_final_chunk(&self) -> bool {
            match self {
                InferenceStep::Prefill {
                    chunk_start_position,
                    chunk_tokens,
                    total_prompt_tokens,
                } => chunk_start_position + chunk_tokens.len() >= *total_prompt_tokens,
                InferenceStep::Decode { .. } => true,
            }
        }

        /// Check if this is a decode step.
        pub fn is_decode(&self) -> bool {
            matches!(self, InferenceStep::Decode { .. })
        }

        /// Get the current KV cache position (cumulative tokens processed).
        pub fn current_position(&self) -> usize {
            match self {
                InferenceStep::Prefill { chunk_start_position, .. } => *chunk_start_position,
                InferenceStep::Decode { position, .. } => *position,
            }
        }

        /// Construct InferenceStep from current input tokens and sequence state.
        ///
        /// Uses cumulative KV cache position to determine prefill vs decode:
        /// - If tokens_processed < total_prompt_tokens: Prefill (building KV cache)
        /// - If tokens_processed >= total_prompt_tokens: Decode (generating)
        pub fn from_sequence(
            input_tokens: Vec<u32>,
            kv_cache_position: usize,
            seq: &Sequence,
        ) -> Self {
            let total_prompt_tokens = seq.prompt_tokens();

            // Decode only if:
            // 1. KV cache position is AT OR PAST the prompt (already processed all prompt tokens)
            // 2. Processing exactly 1 token at a time
            // Otherwise, it's prefill (including when we're completing the last chunk)
            let is_decode = kv_cache_position >= total_prompt_tokens && input_tokens.len() == 1;

            if is_decode {
                InferenceStep::Decode {
                    token: input_tokens.first().copied().unwrap_or(0),
                    position: kv_cache_position,
                }
            } else {
                InferenceStep::Prefill {
                    chunk_tokens: input_tokens,
                    chunk_start_position: kv_cache_position,
                    total_prompt_tokens,
                }
            }
        }
    }

    #[derive(Clone)]
    pub struct ModelInputs {
        pub input_ids: Tensor,
        pub input_ids_full: Option<Tensor>,
        pub seqlen_offsets: Vec<usize>,
        pub seqlen_offsets_full: Option<Vec<usize>>,
        pub context_lens: Vec<(usize, usize)>,
        pub position_ids: Vec<usize>,
        pub paged_attn_meta: Option<PagedAttentionInputMetadata>,
        pub flash_meta: FlashParams,
        pub flash_meta_full: Option<FlashParams>,
        pub request_id: uuid::Uuid,
        pub inference_step: InferenceStep,
    }

    pub struct TextInputsProcessor;

    impl InputsProcessor for TextInputsProcessor {
        fn process_inputs(
            &self,
            _: Option<Arc<Tokenizer>>,
            input_seqs: &mut [&mut Sequence],
            is_prompt: bool,
            is_xlora: bool,
            device: &Device,
            no_kv_cache: bool,
            last_n_context_len: Option<(usize, usize)>,
            return_raw_logits: bool,
            _: Option<Arc<dyn Any>>,
            mut paged_attn_metadata: Option<PagedAttentionMeta>,
            mapper: Option<&dyn DeviceMapper>,
        ) -> Result<InputProcessorOutput> {
            if is_xlora && !is_prompt {
                let prompt = get_prompt_input(
                    input_seqs
                        .iter()
                        .map(|seq| seq.get_toks())
                        .collect::<Vec<_>>(),
                    input_seqs,
                    device,
                    last_n_context_len,
                    return_raw_logits,
                    paged_attn_metadata.as_mut(),
                    mapper,
                )?;
                let completion = get_completion_input(
                    input_seqs
                        .iter()
                        .map(|seq| seq.get_toks())
                        .collect::<Vec<_>>(),
                    input_seqs,
                    device,
                    no_kv_cache,
                    last_n_context_len,
                    return_raw_logits,
                    paged_attn_metadata.as_mut(),
                    mapper,
                )?;
                let InnerInputProcessorOutput {
                    inputs:
                        InputMetadata {
                            input: input_ids_full,
                            positions: seqlen_offsets_full,
                            context_lens: _,
                            position_ids,
                            paged_attn_meta: _,
                            flash_meta: flash_meta_full,
                        },
                    seq_indices,
                } = prompt;
                let InnerInputProcessorOutput {
                    inputs:
                        InputMetadata {
                            input: input_ids,
                            positions: seqlen_offsets,
                            context_lens,
                            position_ids: _,
                            paged_attn_meta,
                            flash_meta,
                        },
                    seq_indices: _,
                } = completion;
                // Compute InferenceStep from current tokens and KV cache position
                let input_tokens: Vec<u32> = input_ids.flatten_all()?.to_vec1()?;
                let kv_position = seqlen_offsets.first().copied().unwrap_or(0);
                let inference_step = InferenceStep::from_sequence(
                    input_tokens,
                    kv_position,
                    input_seqs[0],
                );

                let inputs: Box<dyn Any> = Box::new(ModelInputs {
                    input_ids,
                    input_ids_full: Some(input_ids_full),
                    seqlen_offsets,
                    seqlen_offsets_full: Some(seqlen_offsets_full),
                    context_lens,
                    position_ids,
                    paged_attn_meta,
                    flash_meta,
                    flash_meta_full: Some(flash_meta_full),
                    request_id: input_seqs[0].request_id(),
                    inference_step,
                });
                Ok(InputProcessorOutput {
                    inputs,
                    seq_indices,
                })
            } else if is_xlora && is_prompt {
                let metadata = get_prompt_input(
                    input_seqs
                        .iter()
                        .map(|seq| seq.get_toks())
                        .collect::<Vec<_>>(),
                    input_seqs,
                    device,
                    last_n_context_len,
                    return_raw_logits,
                    paged_attn_metadata.as_mut(),
                    mapper,
                )?;
                let InnerInputProcessorOutput {
                    inputs:
                        InputMetadata {
                            input: input_ids,
                            positions: seqlen_offsets,
                            context_lens,
                            position_ids,
                            paged_attn_meta,
                            flash_meta,
                        },
                    seq_indices,
                } = metadata;

                // Compute InferenceStep from current tokens and KV cache position
                let input_tokens: Vec<u32> = input_ids.flatten_all()?.to_vec1()?;
                let kv_position = seqlen_offsets.first().copied().unwrap_or(0);
                let inference_step = InferenceStep::from_sequence(
                    input_tokens,
                    kv_position,
                    input_seqs[0],
                );

                let inputs: Box<dyn Any> = Box::new(ModelInputs {
                    input_ids: input_ids.clone(),
                    input_ids_full: Some(input_ids),
                    seqlen_offsets: seqlen_offsets.clone(),
                    seqlen_offsets_full: Some(seqlen_offsets),
                    context_lens,
                    position_ids,
                    paged_attn_meta,
                    flash_meta: flash_meta.clone(),
                    flash_meta_full: Some(flash_meta),
                    request_id: input_seqs[0].request_id(),
                    inference_step,
                });
                Ok(InputProcessorOutput {
                    inputs,
                    seq_indices,
                })
            } else if is_prompt {
                let metadata = get_prompt_input(
                    input_seqs
                        .iter()
                        .map(|seq| seq.get_toks())
                        .collect::<Vec<_>>(),
                    input_seqs,
                    device,
                    last_n_context_len,
                    return_raw_logits,
                    paged_attn_metadata.as_mut(),
                    mapper,
                )?;
                let InnerInputProcessorOutput {
                    inputs:
                        InputMetadata {
                            input: input_ids,
                            positions: seqlen_offsets,
                            context_lens,
                            position_ids,
                            paged_attn_meta,
                            flash_meta,
                        },
                    seq_indices,
                } = metadata;

                // NOTE: Some models (e.g. GGUF Phi3) pass token history to distributed PP hooks by
                // flattening `input_ids_full` into a Vec<u32>. Do NOT pad here: padding would inject
                // fake tokens into the history and corrupt sparse KV reconstruction.
                //
                // PP continuation is currently single-sequence; keep this narrowly-scoped.
                let input_ids_full = if input_seqs.len() == 1 {
                    let cpu = Device::Cpu;
                    input_seqs[0]
                        .prefill_prompt_toks()
                        .map(|toks| Tensor::new(toks.to_vec(), &cpu).and_then(|t| t.unsqueeze(0)))
                        .transpose()?
                } else {
                    None
                };

                // Compute InferenceStep from current tokens and KV cache position
                let input_tokens: Vec<u32> = input_ids.flatten_all()?.to_vec1()?;
                let kv_position = seqlen_offsets.first().copied().unwrap_or(0);
                let inference_step = InferenceStep::from_sequence(
                    input_tokens,
                    kv_position,
                    input_seqs[0],
                );

                let inputs: Box<dyn Any> = Box::new(ModelInputs {
                    input_ids,
                    input_ids_full,
                    seqlen_offsets,
                    seqlen_offsets_full: None,
                    context_lens,
                    position_ids,
                    paged_attn_meta,
                    flash_meta,
                    flash_meta_full: None,
                    request_id: input_seqs[0].request_id(),
                    inference_step,
                });
                Ok(InputProcessorOutput {
                    inputs,
                    seq_indices,
                })
            } else {
                let metadata = get_completion_input(
                    input_seqs
                        .iter()
                        .map(|seq| seq.get_toks())
                        .collect::<Vec<_>>(),
                    input_seqs,
                    device,
                    no_kv_cache,
                    last_n_context_len,
                    return_raw_logits,
                    paged_attn_metadata.as_mut(),
                    mapper,
                )?;
                let InnerInputProcessorOutput {
                    inputs:
                        InputMetadata {
                            input: input_ids,
                            positions: seqlen_offsets,
                            context_lens,
                            position_ids,
                            paged_attn_meta,
                            flash_meta,
                        },
                    seq_indices,
                } = metadata;

                // Compute InferenceStep from current tokens and KV cache position
                let input_tokens: Vec<u32> = input_ids.flatten_all()?.to_vec1()?;
                let kv_position = seqlen_offsets.first().copied().unwrap_or(0);
                let inference_step = InferenceStep::from_sequence(
                    input_tokens,
                    kv_position,
                    input_seqs[0],
                );

                let inputs: Box<dyn Any> = Box::new(ModelInputs {
                    input_ids,
                    input_ids_full: None,
                    seqlen_offsets,
                    seqlen_offsets_full: None,
                    context_lens,
                    position_ids,
                    paged_attn_meta,
                    flash_meta,
                    flash_meta_full: None,
                    request_id: input_seqs[0].request_id(),
                    inference_step,
                });
                Ok(InputProcessorOutput {
                    inputs,
                    seq_indices,
                })
            }
        }

        fn get_type(&self) -> InputsProcessorType {
            InputsProcessorType::Text
        }
    }
}
