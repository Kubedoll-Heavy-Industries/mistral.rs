mod default_scheduler;

use std::sync::Arc;

pub use default_scheduler::{DefaultScheduler, DefaultSchedulerMethod, DefaultSchedulerOutput};
use tokio::sync::Mutex;
use uuid::Uuid;

use crate::{
    engine::IntervalLogger,
    paged_attention::{
        BlockEngine, BlockTables, CacheConfig, PagedAttentionScheduler,
        PagedAttentionSchedulerConfig, PagedAttentionSchedulerOutput,
    },
    sequence::Sequence,
};

#[derive(Clone)]
pub enum SchedulerConfig {
    DefaultScheduler {
        method: DefaultSchedulerMethod,
    },
    PagedAttentionMeta {
        max_num_seqs: usize,
        config: CacheConfig,
        /// Maximum tokens per prefill chunk. None = no chunking (default).
        max_prefill_chunk_size: Option<usize>,
    },
}

impl SchedulerConfig {
    pub fn into_scheduler(self) -> Arc<Mutex<dyn Scheduler>> {
        match self {
            Self::DefaultScheduler { method } => {
                Arc::new(Mutex::new(DefaultScheduler::new(method)))
            }
            Self::PagedAttentionMeta {
                max_num_seqs,
                config,
                max_prefill_chunk_size,
            } => Arc::new(Mutex::new(PagedAttentionScheduler::new(
                PagedAttentionSchedulerConfig {
                    max_num_seqs,
                    max_prefill_chunk_size,
                },
                config,
            ))),
        }
    }
}

pub enum SchedulerOutput<'a> {
    DefaultScheduler {
        output: DefaultSchedulerOutput<'a>,
    },
    PagedAttention {
        output: PagedAttentionSchedulerOutput,
    },
}

pub trait Scheduler: Send + Sync {
    fn schedule(&mut self, logger: &IntervalLogger) -> SchedulerOutput<'_>;
    fn waiting_len(&self) -> usize;
    fn running_len(&self) -> usize;
    fn add_seq(&mut self, seq: Sequence);
    /// This may do nothing. It depends on the implementation
    fn free_finished_sequence_groups(&mut self);
    /// Get Mamba state pool indices of finished sequences for freeing.
    /// Called before free_finished_sequence_groups to allow cleanup of hybrid cache slots.
    fn get_finished_mamba_indices(&self) -> Vec<usize>;

    // PagedAttention metadata
    fn block_tables(&self) -> Option<BlockTables>;
    fn block_size(&self) -> Option<usize>;
    fn block_engine(&self) -> Option<Arc<Mutex<BlockEngine>>>;

    /// Set whether prefix caching is enabled. Called by Engine after creation
    /// to synchronize with the global no_prefix_cache setting.
    fn set_prefix_caching_enabled(&mut self, enabled: bool);

    /// Advance prefill chunk offsets for all running sequences in chunked prefill mode.
    /// Called by the engine after processing a batch of prefill chunks.
    /// Default implementation does nothing (for schedulers that don't support chunked prefill).
    fn advance_prefill_chunk_offsets(&mut self) {}

    // Pipeline parallelism support: sequences accessed by request_id (UUID)
    // instead of being batched through schedule().

    /// Check if a PP sequence exists by request_id.
    fn has_pp_sequence(&self, request_id: Uuid) -> bool;

    /// Get a mutable reference to a PP sequence by request_id.
    /// Returns None if not found.
    fn get_pp_sequence_mut(&mut self, request_id: Uuid) -> Option<&mut Sequence>;

    /// Add a PP sequence. Unlike `add_seq()`, PP sequences persist across
    /// multiple forward passes and are accessed directly by request_id.
    fn add_pp_sequence(&mut self, seq: Sequence);

    /// Remove and return a PP sequence by its request_id.
    /// Used for cleanup when a request completes.
    fn remove_pp_sequence(&mut self, request_id: Uuid) -> Option<Sequence>;
}
