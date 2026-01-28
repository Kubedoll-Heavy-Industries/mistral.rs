mod experts;
mod gguf_weight_source;
mod layer;
pub mod routing;
mod weight_source;

use mistralrs_quant::Shard;

// Weight loading infrastructure (some types used directly by models, others for public API)
#[allow(unused_imports)]
pub use gguf_weight_source::{GgufExpertNaming, GgufWeightSource};
#[allow(unused_imports)]
pub use weight_source::{LoadedExpertWeights, MoEWeightSource, QuantProperties, WeightSource};

// Expert container and configuration
pub use experts::{MoEExperts, MoEExpertsConfig};

// MoE layer types (unified infrastructure for all MoE models)
// SoftmaxMoE exported for public API convenience (models often create their own type aliases)
#[allow(unused_imports)]
pub use layer::{GroupLimitedMoE, MoE, MoELayerConfig, MoEOrMlp, SoftmaxMoE, SparseMixerMoE};

// Routing strategies (some used directly via routing submodule, re-exported here for convenience)
#[allow(unused_imports)]
pub use routing::{
    GroupLimitedGreedy, RouteOutput, RoutingConfig, RoutingStrategy, SoftmaxTopK, SparseMixer,
};

pub fn shard(dim: usize, rank: usize, world_size: usize) -> Shard {
    Shard::Simple {
        dim,
        rank,
        world_size,
    }
}
