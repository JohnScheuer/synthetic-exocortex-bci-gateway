#ifndef CUDA_EMBEDDING_ENGINE_CUH
#define CUDA_EMBEDDING_ENGINE_CUH

#include <cuda_runtime.h>

extern "C" {
    void launch_spike_to_embedding_kernel(
        const float* d_spike_rates,
        float* d_embeddings,
        int total_channels,
        int embed_dim,
        cudaStream_t stream
    );
}

#endif // CUDA_EMBEDDING_ENGINE_CUH
