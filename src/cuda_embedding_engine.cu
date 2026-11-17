#include "cuda_embedding_engine.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void spike_to_embedding_kernel(
    const float* __restrict__ d_spike_rates,
    float* __restrict__ d_embeddings,
    int total_channels,
    int embed_dim
) {
    int embed_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (embed_idx < embed_dim) {
        int channels_per_dim = total_channels / embed_dim;
        int start_ch = embed_idx * channels_per_dim;

        float sum = 0.0f;
        for (int i = 0; i < channels_per_dim; ++i) {
            float rate = d_spike_rates[start_ch + i];
            sum += rate * (1.0f + 0.01f * (i % 8));
        }

        d_embeddings[embed_idx] = tanhf(sum * 0.05f) * 20.0f;
    }
}

extern "C" {
    void launch_spike_to_embedding_kernel(
        const float* d_spike_rates,
        float* d_embeddings,
        int total_channels,
        int embed_dim,
        cudaStream_t stream
    ) {
        int threads_per_block = 256;
        int blocks_per_grid = (embed_dim + threads_per_block - 1) / threads_per_block;

        spike_to_embedding_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(
            d_spike_rates,
            d_embeddings,
            total_channels,
            embed_dim
        );
    }
}
