#ifndef CUDA_EMBEDDING_ENGINE_CUH
#define CUDA_EMBEDDING_ENGINE_CUH

#include <cuda_runtime.h>

class CUDAEmbeddingEngine {
private:
    float* d_rates_;
    float* d_embeddings_;

public:
    CUDAEmbeddingEngine();
    ~CUDAEmbeddingEngine();

    void launch_embedding_kernel(const float* h_rates, float* h_embeddings);
};

#endif
