#include "cuda_embedding_engine.cuh"
#include "avx2_signal_decoder.hpp"
#include "cognitive_orchestrator.hpp"
#include <iostream>

__global__ void semantic_embedding_kernel(const float* __restrict__ rates, float* __restrict__ embeddings, int channels, int embed_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < embed_dim) {
        float sum = 0.0f;
        int stride = channels / embed_dim;
        for (int i = 0; i < stride; ++i) {
            int chan_idx = idx * stride + i;
            float r = rates[chan_idx];
            sum += r * __sinf(chan_idx * 0.01f + idx * 0.05f);
        }
        embeddings[idx] = __tanf(sum * 0.001f);
    }
}

CUDAEmbeddingEngine::CUDAEmbeddingEngine() {
    cudaMalloc(&d_rates_, CHANNELS * sizeof(float));
    cudaMalloc(&d_embeddings_, EMBEDDING_DIM * sizeof(float));
}

CUDAEmbeddingEngine::~CUDAEmbeddingEngine() {
    cudaFree(d_rates_);
    cudaFree(d_embeddings_);
}

void CUDAEmbeddingEngine::launch_embedding_kernel(const float* h_rates, float* h_embeddings) {
    cudaMemcpy(d_rates_, h_rates, CHANNELS * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (EMBEDDING_DIM + threads - 1) / threads;

    semantic_embedding_kernel<<<blocks, threads>>>(d_rates_, d_embeddings_, CHANNELS, EMBEDDING_DIM);
    cudaDeviceSynchronize();

    cudaMemcpy(h_embeddings, d_embeddings_, EMBEDDING_DIM * sizeof(float), cudaMemcpyDeviceToHost);
}
