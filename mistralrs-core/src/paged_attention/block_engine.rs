use std::{
    collections::HashMap,
    hash::Hash,
    marker::PhantomData,
    ops::Deref,
    sync::{Arc, Mutex, MutexGuard},
};

use super::block_engine_sequence::BlockEngineSequence;
use super::prefix_cacher::PrefixCacher;

#[derive(Debug, Clone)]
pub struct LogicalTokenBlock {
    tokens: Vec<usize>,
    block_size: usize,
    num_tokens: usize,
}

impl LogicalTokenBlock {
    pub fn new(block_size: usize) -> Self {
        Self {
            tokens: [0].repeat(block_size),
            block_size,
            num_tokens: 0,
        }
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    pub fn num_tokens(&self) -> usize {
        self.num_tokens
    }

    pub fn is_full(&self) -> bool {
        self.num_tokens == self.block_size
    }

    pub fn is_empty(&self) -> bool {
        self.num_tokens == 0
    }

    pub fn append_token_id(&mut self, token: usize) {
        assert!(!self.is_full());
        self.tokens[self.num_tokens] = token;
        self.num_tokens += 1;
    }

    pub fn pop_token(&mut self) {
        assert_ne!(self.num_tokens, 0);
        self.tokens.pop();
        self.num_tokens -= 1;
    }

    pub fn toks(&self) -> &[usize] {
        &self.tokens
    }
}

impl Hash for LogicalTokenBlock {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.tokens.hash(state);
    }
}

#[derive(Hash, PartialEq, Eq)]
pub struct _PhysicalTokenBlock {
    pub block_id: usize,
    block_size: usize,
    refcount: usize,
    is_gpu: bool,
}

impl _PhysicalTokenBlock {
    pub fn refcount(&self) -> usize {
        self.refcount
    }
    pub fn increment_refcount(&mut self) {
        self.refcount += 1;
    }
    pub fn decrement_refcount(&mut self) {
        assert!(self.refcount >= 1);
        self.refcount -= 1;
    }
}

pub struct PhysicalTokenBlock(pub Mutex<_PhysicalTokenBlock>);

impl std::fmt::Debug for PhysicalTokenBlock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.0.lock() {
            Ok(inner) => f
                .debug_struct("PhysicalTokenBlock")
                .field("block_id", &inner.block_id)
                .field("block_size", &inner.block_size)
                .field("refcount", &inner.refcount)
                .field("is_gpu", &inner.is_gpu)
                .finish(),
            Err(_) => write!(f, "PhysicalTokenBlock(<locked>)"),
        }
    }
}

impl PhysicalTokenBlock {
    pub fn deref_mut(&self) -> MutexGuard<'_, _PhysicalTokenBlock> {
        loop {
            if let Ok(v) = self.0.try_lock() {
                return v;
            }
        }
    }
}

impl PartialEq for PhysicalTokenBlock {
    fn eq(&self, other: &Self) -> bool {
        *self.deref_mut() == *other.deref_mut()
    }
}

impl Hash for PhysicalTokenBlock {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.deref_mut().hash(state)
    }
}

impl Eq for PhysicalTokenBlock {}

type BlockTable = Vec<Arc<PhysicalTokenBlock>>;
struct GPUAllocator;

struct GPUAllocatorWrapper(usize);
impl Deref for GPUAllocatorWrapper {
    type Target = usize;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

struct Allocator<T> {
    free_blocks: BlockTable,
    _ghost: PhantomData<T>,
}

impl<T> Allocator<T> {
    fn allocate(&mut self) -> Arc<PhysicalTokenBlock> {
        let block = self.free_blocks.pop().unwrap();
        block.deref_mut().refcount = 1;
        block
    }

    fn free_block(&mut self, block: Arc<PhysicalTokenBlock>) {
        if block.deref_mut().refcount == 0 {
            panic!(
                "PhysicalTokenBlock with id {} experienced a double free!",
                block.deref_mut().block_id
            );
        }
        block.deref_mut().refcount -= 1;
        if block.deref_mut().refcount == 0 {
            self.free_blocks.push(block);
        }
    }
}

impl Allocator<GPUAllocator> {
    fn new(block_size: usize, num_blocks: usize) -> Self {
        let mut free_blocks = Vec::new();
        for id in 0..num_blocks {
            free_blocks.push(Arc::new(PhysicalTokenBlock(Mutex::new(
                _PhysicalTokenBlock {
                    block_id: id,
                    block_size,
                    refcount: 0,
                    is_gpu: true,
                },
            ))))
        }
        Allocator {
            free_blocks,
            _ghost: PhantomData,
        }
    }

    fn get_num_free_blocks(&self) -> GPUAllocatorWrapper {
        GPUAllocatorWrapper(self.free_blocks.len())
    }
}

#[derive(Debug)]
pub enum AllocStatus {
    Ok,
    Later { waitlisted_count: usize },
    Impossible,
}

type SeqID = usize;

/// A BlockEngine maps each Sequence (identified by its SeqID), to physical token blocks.
/// The physical token blocks may not match the logical token blocks because during
/// scheduling, physical blocks are allocated to accommodate the new tokens generated.
/// These new tokens will be added to the logical token block for each sequence.
pub struct BlockEngine {
    num_gpu_blocks: usize,
    block_size: usize,
    gpu_allocator: Allocator<GPUAllocator>,
    pub block_tables: HashMap<SeqID, BlockTable>,
    /// Prefix cache for reusing KV cache blocks across requests with shared prefixes.
    prefix_cacher: PrefixCacher,
    /// Track number of cached blocks used per sequence (for freeing).
    cached_blocks_per_seq: HashMap<SeqID, usize>,
}

pub type BlockTables = HashMap<usize, BlockTable>;

impl BlockEngine {
    #[must_use]
    pub fn new(block_size: usize, num_gpu_blocks: usize, prefix_caching_enabled: bool) -> Self {
        Self {
            num_gpu_blocks,
            block_size,
            gpu_allocator: Allocator::<GPUAllocator>::new(block_size, num_gpu_blocks),
            block_tables: HashMap::new(),
            prefix_cacher: PrefixCacher::new(prefix_caching_enabled),
            cached_blocks_per_seq: HashMap::new(),
        }
    }

    /// Check if prefix caching is enabled.
    pub fn prefix_caching_enabled(&self) -> bool {
        self.prefix_cacher.is_enabled()
    }

    /// Set whether prefix caching is enabled.
    pub fn set_prefix_caching_enabled(&mut self, enabled: bool) {
        self.prefix_cacher.set_enabled(enabled);
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    pub fn can_allocate(&mut self, seq: &mut impl BlockEngineSequence) -> AllocStatus {
        let logical_blocks = seq.logical_token_blocks();
        let num_required_blocks = logical_blocks.len();

        // Check how many blocks we can get from prefix cache
        let num_cached = if self.prefix_cacher.is_enabled() {
            let (_, num_matched) = self.prefix_cacher.match_prefix(logical_blocks);
            num_matched
        } else {
            0
        };

        // We only need to allocate blocks that aren't in the cache
        let num_new_blocks_needed = num_required_blocks.saturating_sub(num_cached);
        let num_free_gpu_blocks = self.gpu_allocator.get_num_free_blocks();

        // Also count evictable blocks from prefix cache as potentially available
        let num_evictable = self.prefix_cacher.num_evictable_blocks();
        let total_available = *num_free_gpu_blocks + num_evictable;

        if self.num_gpu_blocks < num_required_blocks {
            AllocStatus::Impossible
        } else if total_available < num_new_blocks_needed {
            AllocStatus::Later {
                waitlisted_count: seq.increment_waitlist_count(),
            }
        } else {
            AllocStatus::Ok
        }
    }

    pub fn allocate(&mut self, seq: &mut impl BlockEngineSequence) {
        let num_blocks_needed = seq.logical_token_blocks().len();
        let seq_id = seq.get_id();
        let block_size = seq.block_size();

        // If there are prefill physical blocks, use those here.
        if let Some(physical_blocks_prefill) = seq.take_physical_blocks_prefill() {
            let mut block_table = physical_blocks_prefill.clone();
            let n_extra_blocks = num_blocks_needed - block_table.len();
            for _ in 0..n_extra_blocks {
                block_table.push(self.allocate_block_with_eviction());
            }
            self.block_tables.insert(seq_id, block_table);
            self.cached_blocks_per_seq.insert(seq_id, 0);
            seq.set_prefix_cache_len(0);
            return;
        }

        // Re-borrow logical_blocks after the mutable borrow above is done
        let logical_blocks = seq.logical_token_blocks();

        // Try to get blocks from prefix cache
        let (cached_blocks, num_cached) = if self.prefix_cacher.is_enabled() {
            self.prefix_cacher.match_prefix(logical_blocks)
        } else {
            (Vec::new(), 0)
        };

        let mut block_table = Vec::with_capacity(num_blocks_needed);

        // Use cached blocks for the prefix
        for (idx, physical_block) in cached_blocks {
            // Extend block_table to the right size
            while block_table.len() < idx {
                block_table.push(self.allocate_block_with_eviction());
            }
            // The cached block already has its refcount incremented by match_prefix
            block_table.push(physical_block);
        }

        // Allocate new blocks for the rest
        for _ in block_table.len()..num_blocks_needed {
            block_table.push(self.allocate_block_with_eviction());
        }

        self.cached_blocks_per_seq.insert(seq_id, num_cached);
        self.block_tables.insert(seq_id, block_table);

        // Calculate number of cached tokens (full blocks only)
        // num_cached is the number of full blocks that were cache hits
        let cached_tokens = num_cached * block_size;
        seq.set_prefix_cache_len(cached_tokens);
    }

    /// Check if the last allocate() call resulted in a prefix cache hit.
    /// Returns the number of blocks that were reused from cache.
    pub fn last_allocate_had_cache_hit(&self, seq_id: usize) -> usize {
        self.cached_blocks_per_seq
            .get(&seq_id)
            .copied()
            .unwrap_or(0)
    }

    /// Allocate a block, evicting from prefix cache if necessary.
    fn allocate_block_with_eviction(&mut self) -> Arc<PhysicalTokenBlock> {
        // Try to allocate from free pool first
        if *self.gpu_allocator.get_num_free_blocks() > 0 {
            return self.gpu_allocator.allocate();
        }

        // Need to evict from prefix cache
        let evicted = self.prefix_cacher.evict_blocks(1);
        for block in evicted {
            // Decrement refcount and return to free pool
            block.deref_mut().decrement_refcount();
            if block.deref_mut().refcount == 0 {
                self.gpu_allocator.free_blocks.push(block);
            }
        }

        // Now allocate
        self.gpu_allocator.allocate()
    }

    pub fn can_append_token_to_seq(&self, seq: &impl BlockEngineSequence) -> bool {
        let free_blocks = self.gpu_allocator.get_num_free_blocks();
        let evictable = self.prefix_cacher.num_evictable_blocks();
        // Physical blocks = logical blocks
        seq.blocks_to_add_new_tok() <= *free_blocks + evictable
    }

    /// Free a sequence's blocks and optionally cache them for prefix reuse.
    /// If `logical_blocks` is provided and prefix caching is enabled, full blocks
    /// will be added to the prefix cache for potential reuse by future requests.
    pub fn free_sequence_with_caching(
        &mut self,
        id: usize,
        logical_blocks: Option<&[LogicalTokenBlock]>,
    ) {
        // Handle double free if run out of tokens
        if let Some(block_table) = self.block_tables.remove(&id) {
            let num_cached = self.cached_blocks_per_seq.remove(&id).unwrap_or(0);

            // Cache blocks for prefix reuse (skip already-cached blocks)
            if let Some(logical_blocks) = logical_blocks {
                if self.prefix_cacher.is_enabled() && block_table.len() == logical_blocks.len() {
                    // Insert new blocks into cache (starting after cached blocks)
                    self.prefix_cacher
                        .insert_blocks(logical_blocks, &block_table, num_cached);
                }
            }

            // Release cached blocks' reference counts
            if num_cached > 0 {
                if let Some(logical_blocks) = logical_blocks {
                    let cached_logical = &logical_blocks[..num_cached.min(logical_blocks.len())];
                    self.prefix_cacher.release_blocks(cached_logical);
                }
            }

            // Free all blocks - each block needs its sequence reference released.
            // The cache holds its own reference (incremented in insert_blocks),
            // so we must release the sequence's reference for proper refcounting.
            for block in block_table.iter() {
                self.gpu_allocator.free_block(block.clone());
            }
        }
    }

    pub fn free_sequence(&mut self, id: usize) {
        // Free without caching (for aborted sequences or when we don't have logical blocks)
        if let Some(block_table) = self.block_tables.remove(&id) {
            self.cached_blocks_per_seq.remove(&id);
            // Free all blocks
            for block in block_table.iter() {
                self.gpu_allocator.free_block(block.clone());
            }
        }
    }

    /// Free a sequence's blocks during preemption.
    /// This properly releases prefix cache refs so cached blocks can be evicted.
    pub fn free_sequence_for_preemption(
        &mut self,
        id: usize,
        logical_blocks: &[LogicalTokenBlock],
    ) {
        if let Some(block_table) = self.block_tables.remove(&id) {
            let num_cached = self.cached_blocks_per_seq.remove(&id).unwrap_or(0);

            // Release cached blocks' reference counts in the prefix cache
            if num_cached > 0 && self.prefix_cacher.is_enabled() {
                let cached_logical = &logical_blocks[..num_cached.min(logical_blocks.len())];
                self.prefix_cacher.release_blocks(cached_logical);
            }

            // Free all blocks
            for block in block_table.iter() {
                self.gpu_allocator.free_block(block.clone());
            }
        }
    }

    // Returns the COW mapping (src, dst).
    // COW is performed if there are multiple references to the last physical block.
    pub fn append_token_slot_to_seq(
        &mut self,
        sequence: &impl BlockEngineSequence,
    ) -> Option<(usize, usize)> {
        let seq_id = sequence.get_id();
        let blocks_to_add = sequence.blocks_to_add_new_tok();

        // Check if table exists
        if !self.block_tables.contains_key(&seq_id) {
            return None;
        }

        match blocks_to_add {
            1 => {
                // Allocate first, then push to table
                let new_block = self.allocate_block_with_eviction();
                self.block_tables.get_mut(&seq_id).unwrap().push(new_block);
                None
            }
            0 => {
                // Get the last block's info first
                let table = self.block_tables.get(&seq_id).unwrap();
                let last_block = table.last().unwrap();
                let is_gpu = last_block.deref_mut().is_gpu;
                let refcount = last_block.deref_mut().refcount;

                assert!(is_gpu);

                if refcount == 1 {
                    None
                } else {
                    // We would be writing into shared, so COW.
                    let old_block = last_block.clone();
                    let old_number = old_block.deref_mut().block_id;

                    // Now allocate and mutate
                    let new_block = self.allocate_block_with_eviction();
                    let new_number = new_block.deref_mut().block_id;

                    // Free old block
                    self.gpu_allocator.free_block(old_block);

                    // Replace in table
                    let table = self.block_tables.get_mut(&seq_id).unwrap();
                    *table.last_mut().unwrap() = new_block;

                    Some((old_number, new_number))
                }
            }
            _ => {
                unreachable!()
            }
        }
    }

    /// Get prefix cache statistics (hits, misses).
    pub fn prefix_cache_stats(&self) -> (usize, usize) {
        self.prefix_cacher.stats()
    }

    /// Get prefix cache hit rate as a percentage.
    pub fn prefix_cache_hit_rate(&self) -> f64 {
        self.prefix_cacher.hit_rate()
    }

    /// Get number of blocks in prefix cache.
    pub fn prefix_cache_size(&self) -> usize {
        self.prefix_cacher.num_cached_blocks()
    }

    /// Get number of free blocks in the GPU allocator.
    #[cfg(test)]
    pub fn num_free_blocks(&self) -> usize {
        *self.gpu_allocator.get_num_free_blocks()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal sequence implementation for testing block engine allocation.
    struct MockSequence {
        id: usize,
        logical_blocks: Vec<LogicalTokenBlock>,
        block_size: usize,
        waitlist_count: usize,
        prefix_cache_len: usize,
        physical_blocks_prefill: Option<Vec<Arc<PhysicalTokenBlock>>>,
    }

    impl MockSequence {
        fn new(id: usize, block_size: usize, num_tokens: usize) -> Self {
            Self::new_with_token_offset(id, block_size, num_tokens, 0)
        }

        /// Create a sequence with tokens starting from a given offset.
        /// This allows creating sequences with different token values for cache testing.
        fn new_with_token_offset(id: usize, block_size: usize, num_tokens: usize, token_offset: usize) -> Self {
            let num_blocks = (num_tokens + block_size - 1) / block_size;
            let mut logical_blocks = Vec::with_capacity(num_blocks);

            let mut token_id = token_offset;
            let mut remaining = num_tokens;
            for _ in 0..num_blocks {
                let mut block = LogicalTokenBlock::new(block_size);
                let tokens_in_block = remaining.min(block_size);
                for _ in 0..tokens_in_block {
                    block.append_token_id(token_id);
                    token_id += 1;
                }
                logical_blocks.push(block);
                remaining = remaining.saturating_sub(block_size);
            }

            Self {
                id,
                logical_blocks,
                block_size,
                waitlist_count: 0,
                prefix_cache_len: 0,
                physical_blocks_prefill: None,
            }
        }

        fn with_full_blocks(id: usize, block_size: usize, num_blocks: usize) -> Self {
            Self::new(id, block_size, num_blocks * block_size)
        }

        fn with_full_blocks_offset(id: usize, block_size: usize, num_blocks: usize, token_offset: usize) -> Self {
            Self::new_with_token_offset(id, block_size, num_blocks * block_size, token_offset)
        }
    }

    impl BlockEngineSequence for MockSequence {
        fn blocks_to_add_new_tok(&self) -> usize {
            if let Some(last) = self.logical_blocks.last() {
                if last.is_full() { 1 } else { 0 }
            } else {
                1
            }
        }

        fn take_physical_blocks_prefill(&mut self) -> Option<Vec<Arc<PhysicalTokenBlock>>> {
            self.physical_blocks_prefill.take()
        }

        fn get_id(&self) -> usize {
            self.id
        }

        fn logical_token_blocks(&self) -> &[LogicalTokenBlock] {
            &self.logical_blocks
        }

        fn increment_waitlist_count(&mut self) -> usize {
            let prev = self.waitlist_count;
            self.waitlist_count += 1;
            prev
        }

        fn set_prefix_cache_len(&mut self, len: usize) {
            self.prefix_cache_len = len;
        }

        fn block_size(&self) -> usize {
            self.block_size
        }
    }

    // ========================================================================
    // Basic allocation tests
    // ========================================================================

    #[test]
    fn test_basic_allocation() {
        let block_size = 16;
        let num_gpu_blocks = 10;
        let mut engine = BlockEngine::new(block_size, num_gpu_blocks, false);

        let mut seq = MockSequence::with_full_blocks(1, block_size, 2);

        assert!(matches!(engine.can_allocate(&mut seq), AllocStatus::Ok));
        engine.allocate(&mut seq);

        assert!(engine.block_tables.contains_key(&1));
        assert_eq!(engine.block_tables.get(&1).unwrap().len(), 2);
        assert_eq!(engine.num_free_blocks(), num_gpu_blocks - 2);
    }

    #[test]
    fn test_free_sequence() {
        let block_size = 16;
        let num_gpu_blocks = 10;
        let mut engine = BlockEngine::new(block_size, num_gpu_blocks, false);

        let mut seq = MockSequence::with_full_blocks(1, block_size, 3);
        engine.allocate(&mut seq);

        assert_eq!(engine.num_free_blocks(), num_gpu_blocks - 3);

        engine.free_sequence(1);

        assert_eq!(engine.num_free_blocks(), num_gpu_blocks);
        assert!(!engine.block_tables.contains_key(&1));
    }

    #[test]
    fn test_allocation_impossible_when_sequence_too_large() {
        let block_size = 16;
        let num_gpu_blocks = 5;
        let mut engine = BlockEngine::new(block_size, num_gpu_blocks, false);

        // Sequence needs 10 blocks, but only 5 available
        let mut seq = MockSequence::with_full_blocks(1, block_size, 10);

        assert!(matches!(engine.can_allocate(&mut seq), AllocStatus::Impossible));
    }

    #[test]
    fn test_allocation_later_when_insufficient_blocks() {
        let block_size = 16;
        let num_gpu_blocks = 10;
        let mut engine = BlockEngine::new(block_size, num_gpu_blocks, false);

        // Allocate 8 blocks
        let mut seq1 = MockSequence::with_full_blocks(1, block_size, 8);
        engine.allocate(&mut seq1);

        // Try to allocate 5 more (only 2 free)
        let mut seq2 = MockSequence::with_full_blocks(2, block_size, 5);
        assert!(matches!(engine.can_allocate(&mut seq2), AllocStatus::Later { .. }));
    }

    // ========================================================================
    // Block exhaustion bug reproduction
    // ========================================================================

    #[test]
    fn test_exhaustion_without_prefix_cache() {
        let block_size = 16;
        let num_gpu_blocks = 4;
        let mut engine = BlockEngine::new(block_size, num_gpu_blocks, false);

        // Allocate all blocks
        let mut seq1 = MockSequence::with_full_blocks(1, block_size, 4);
        engine.allocate(&mut seq1);

        assert_eq!(engine.num_free_blocks(), 0);

        // Trying to allocate more should return Later, not panic
        let mut seq2 = MockSequence::with_full_blocks(2, block_size, 1);
        let status = engine.can_allocate(&mut seq2);
        assert!(matches!(status, AllocStatus::Later { .. }));
    }

    #[test]
    fn test_exhaustion_with_prefix_cache_evictable() {
        let block_size = 16;
        let num_gpu_blocks = 4;
        let mut engine = BlockEngine::new(block_size, num_gpu_blocks, true);

        // Allocate all blocks to a sequence
        let mut seq1 = MockSequence::with_full_blocks(1, block_size, 4);
        engine.allocate(&mut seq1);

        assert_eq!(engine.num_free_blocks(), 0);

        // Free with caching - blocks go to prefix cache
        let logical_blocks = seq1.logical_blocks.clone();
        engine.free_sequence_with_caching(1, Some(&logical_blocks));

        // Blocks are now in prefix cache with ref_count = 0 (evictable)
        // (The cache holds them but the CachedBlockEntry.ref_count is 0)

        // Create a new sequence with DIFFERENT tokens that won't match cache
        // Use token_offset=10000 so tokens are completely different
        let mut seq2 = MockSequence::with_full_blocks_offset(2, block_size, 2, 10000);

        // This should be able to evict from cache and allocate
        let status = engine.can_allocate(&mut seq2);
        assert!(matches!(status, AllocStatus::Ok), "Expected Ok, got {:?}", status);

        // Allocation should succeed (evicting from cache)
        engine.allocate(&mut seq2);
        assert!(engine.block_tables.contains_key(&2));
    }

    /// This test reproduces the panic scenario: all blocks are held by active
    /// sequences with ref_count > 0, eviction fails, and allocate() panics.
    #[test]
    fn test_exhaustion_with_all_blocks_actively_referenced() {
        let block_size = 16;
        let num_gpu_blocks = 4;
        let mut engine = BlockEngine::new(block_size, num_gpu_blocks, true);

        // Allocate all blocks across two sequences
        let mut seq1 = MockSequence::with_full_blocks(1, block_size, 2);
        let mut seq2 = MockSequence::with_full_blocks(2, block_size, 2);

        engine.allocate(&mut seq1);
        engine.allocate(&mut seq2);

        assert_eq!(engine.num_free_blocks(), 0);

        // Both sequences are still "running" - blocks have refcount = 1
        // There's nothing in the prefix cache LRU queue to evict

        // Trying to allocate more should NOT panic
        let mut seq3 = MockSequence::with_full_blocks(3, block_size, 1);
        let status = engine.can_allocate(&mut seq3);

        // Should return Later, not Ok (and definitely not panic)
        assert!(
            matches!(status, AllocStatus::Later { .. }),
            "Expected Later when all blocks are in use, got {:?}",
            status
        );
    }

    /// Test that simulates the deterioration scenario: a sequence holds cache
    /// refs that prevent eviction, leading to allocation failure.
    #[test]
    fn test_prefix_cache_ref_leak_scenario() {
        let block_size = 16;
        let num_gpu_blocks = 6;
        let mut engine = BlockEngine::new(block_size, num_gpu_blocks, true);

        // First sequence: allocate and complete (adds blocks to cache)
        let mut seq1 = MockSequence::with_full_blocks(1, block_size, 3);
        engine.allocate(&mut seq1);
        let logical_blocks1 = seq1.logical_blocks.clone();
        engine.free_sequence_with_caching(1, Some(&logical_blocks1));

        // At this point: 3 blocks in cache (evictable), 3 free blocks
        assert_eq!(engine.prefix_cache_size(), 3);

        // Second sequence: same prefix, gets cache hit, is still running
        let mut seq2 = MockSequence::with_full_blocks(2, block_size, 3);
        // Use same tokens to get cache hit
        seq2.logical_blocks = logical_blocks1.clone();
        engine.allocate(&mut seq2);

        // seq2 now holds refs to cached blocks (refcount incremented by match_prefix)
        // cached_blocks_per_seq should show 3 blocks were from cache
        assert_eq!(engine.last_allocate_had_cache_hit(2), 3);

        // Allocate remaining free blocks with a DIFFERENT sequence (different tokens)
        let mut seq3 = MockSequence::with_full_blocks_offset(3, block_size, 3, 10000);
        engine.allocate(&mut seq3);

        assert_eq!(engine.num_free_blocks(), 0);

        // Now try to allocate more - this requires evicting from cache,
        // but cached blocks have ref_count > 0 due to seq2 (it matched the cache)
        let mut seq4 = MockSequence::with_full_blocks_offset(4, block_size, 1, 20000);
        let status = engine.can_allocate(&mut seq4);

        // Should be Later (evictable = 0 because cached blocks are referenced by seq2)
        assert!(
            matches!(status, AllocStatus::Later { .. }),
            "Expected Later when cached blocks are referenced, got {:?}",
            status
        );
    }

    // ========================================================================
    // Refcount correctness tests
    // ========================================================================

    #[test]
    fn test_refcount_after_allocation() {
        let block_size = 16;
        let num_gpu_blocks = 10;
        let mut engine = BlockEngine::new(block_size, num_gpu_blocks, false);

        let mut seq = MockSequence::with_full_blocks(1, block_size, 2);
        engine.allocate(&mut seq);

        // All allocated blocks should have refcount = 1
        for block in engine.block_tables.get(&1).unwrap() {
            assert_eq!(block.deref_mut().refcount, 1);
        }
    }

    #[test]
    fn test_refcount_after_free() {
        let block_size = 16;
        let num_gpu_blocks = 10;
        let mut engine = BlockEngine::new(block_size, num_gpu_blocks, false);

        let mut seq = MockSequence::with_full_blocks(1, block_size, 2);
        engine.allocate(&mut seq);
        engine.free_sequence(1);

        // All free blocks should have refcount = 0
        for block in &engine.gpu_allocator.free_blocks {
            assert_eq!(block.deref_mut().refcount, 0);
        }
    }

    #[test]
    fn test_double_free_panics() {
        let block_size = 16;
        let num_gpu_blocks = 10;
        let mut engine = BlockEngine::new(block_size, num_gpu_blocks, false);

        let mut seq = MockSequence::with_full_blocks(1, block_size, 2);
        engine.allocate(&mut seq);

        // First free succeeds
        engine.free_sequence(1);

        // Second free should be a no-op (block_tables no longer contains id)
        // This tests that we handle double-free gracefully
        engine.free_sequence(1); // Should not panic
    }

    /// Verify that free_sequence_with_caching properly releases the sequence's
    /// reference while letting the cache hold its own reference.
    #[test]
    fn test_refcount_correct_after_free_with_caching() {
        let block_size = 16;
        let num_gpu_blocks = 4;
        let mut engine = BlockEngine::new(block_size, num_gpu_blocks, true);

        // Allocate blocks
        let mut seq = MockSequence::with_full_blocks(1, block_size, 2);
        engine.allocate(&mut seq);

        // After allocation, blocks have refcount = 1 (held by sequence)
        for block in engine.block_tables.get(&1).unwrap() {
            assert_eq!(block.deref_mut().refcount, 1, "Block should have refcount 1 after allocation");
        }

        // Free with caching
        let logical_blocks = seq.logical_blocks.clone();
        engine.free_sequence_with_caching(1, Some(&logical_blocks));

        // Blocks should now be in cache with refcount = 1 (cache's reference only)
        // The sequence's reference should have been released
        assert_eq!(engine.prefix_cache_size(), 2);

        // Verify we can evict and the refcount is correct
        // Allocate a different sequence to force eviction
        let mut seq2 = MockSequence::with_full_blocks_offset(2, block_size, 4, 10000);
        let status = engine.can_allocate(&mut seq2);
        assert!(matches!(status, AllocStatus::Ok), "Should be able to allocate by evicting");

        engine.allocate(&mut seq2);

        // All 4 blocks should now be allocated to seq2
        assert_eq!(engine.num_free_blocks(), 0);
        assert!(engine.block_tables.contains_key(&2));
        assert_eq!(engine.block_tables.get(&2).unwrap().len(), 4);
    }

    /// Test that verifies the fix from cagyirey/fix/paged-attention-prefix-cache-refcount:
    /// When a sequence matches blocks from the prefix cache, the physical block's
    /// refcount must be incremented so both cache and sequence hold valid refs.
    /// Without this fix, the sequence could free a block that the cache still references.
    #[test]
    fn test_cache_match_increments_physical_refcount() {
        let block_size = 16;
        let num_gpu_blocks = 6;
        let mut engine = BlockEngine::new(block_size, num_gpu_blocks, true);

        // First sequence: allocate and complete (adds blocks to cache)
        let mut seq1 = MockSequence::with_full_blocks(1, block_size, 2);
        engine.allocate(&mut seq1);
        let logical_blocks1 = seq1.logical_blocks.clone();
        engine.free_sequence_with_caching(1, Some(&logical_blocks1));

        // Blocks are now in cache with refcount = 1 (cache's ref)
        assert_eq!(engine.prefix_cache_size(), 2);

        // Second sequence: same prefix, should get cache hit
        let mut seq2 = MockSequence::with_full_blocks(2, block_size, 2);
        seq2.logical_blocks = logical_blocks1.clone();
        engine.allocate(&mut seq2);

        // Verify cache hit
        assert_eq!(engine.last_allocate_had_cache_hit(2), 2);

        // The physical blocks should now have refcount = 2:
        // - 1 from the cache
        // - 1 from seq2
        for block in engine.block_tables.get(&2).unwrap() {
            assert_eq!(
                block.deref_mut().refcount, 2,
                "Block should have refcount 2 (cache + sequence)"
            );
        }

        // Free seq2 - this should decrement refcount to 1, NOT free to pool
        let logical_blocks2 = seq2.logical_blocks.clone();
        engine.free_sequence_with_caching(2, Some(&logical_blocks2));

        // Cache should still hold the blocks (refcount = 1)
        assert_eq!(engine.prefix_cache_size(), 2);

        // Blocks should NOT be in the free pool yet
        // (they're still referenced by cache)
        assert_eq!(engine.num_free_blocks(), 4); // 6 total - 2 in cache = 4 free
    }

    /// Test the specific scenario from the production bug:
    /// Multiple sequences complete, blocks accumulate in cache with wrong refcount,
    /// eventually causing allocation failure.
    #[test]
    fn test_multiple_sequences_with_caching_no_leak() {
        let block_size = 16;
        let num_gpu_blocks = 8;
        let mut engine = BlockEngine::new(block_size, num_gpu_blocks, true);

        // Simulate multiple requests completing
        for i in 0..10 {
            let mut seq = MockSequence::with_full_blocks_offset(i, block_size, 2, i * 1000);

            let status = engine.can_allocate(&mut seq);
            assert!(matches!(status, AllocStatus::Ok),
                "Iteration {}: Expected Ok, got {:?}", i, status);

            engine.allocate(&mut seq);

            // Complete the sequence
            let logical_blocks = seq.logical_blocks.clone();
            engine.free_sequence_with_caching(i, Some(&logical_blocks));
        }

        // After 10 iterations, cache should have evicted old blocks
        // and we should still be able to allocate
        let mut final_seq = MockSequence::with_full_blocks_offset(100, block_size, 4, 100000);
        let status = engine.can_allocate(&mut final_seq);
        assert!(matches!(status, AllocStatus::Ok),
            "Final allocation should succeed, got {:?}", status);

        engine.allocate(&mut final_seq);
        assert!(engine.block_tables.contains_key(&100));
    }
}
