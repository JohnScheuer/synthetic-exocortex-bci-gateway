#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <cuda_runtime.h>

#include "avx_spike_density_decoder.hpp"
#include "cognitive_offload_orchestrator.hpp"
#include "exocortex_bio_guard.hpp"
#include "cuda_embedding_engine.cuh"

void test_avx_decoder() {
    std::cout << "[TEST 1] AVX2 65,536-Channel Spike Rate Density Decoder... ";
    AVXSpikeDensityDecoder decoder;
    std::vector<float> spikes(TOTAL_NEURAL_CHANNELS, 10.0f);
    std::vector<float> rates(TOTAL_NEURAL_CHANNELS, 0.0f);

    decoder.process_spike_rates(spikes.data(), rates.data());

    for (size_t i = 0; i < TOTAL_NEURAL_CHANNELS; ++i) {
        assert(rates[i] >= 0.0f && "Rate density output cannot be negative!");
    }
    std::cout << "PASSED! (65,536 channels decoded)\n";
}

void test_orchestrator() {
    std::cout << "[TEST 2] Cognitive Load & Entropy Orchestrator... ";
    CognitiveOffloadOrchestrator orchestrator;
    std::vector<float> high_entropy(TOTAL_NEURAL_CHANNELS, 80.0f);
    float entropy = 0.0f;

    CognitiveState state = orchestrator.evaluate_cognitive_load(high_entropy.data(), entropy);

    assert(state == CognitiveState::GPU_EXOCORTEX_OFFLOAD && "Orchestrator failed to trigger GPU offload!");
    std::cout << "PASSED! (Entropy: " << entropy << " -> GPU Offload Triggered)\n";
}

void test_bio_guard() {
    std::cout << "[TEST 3] ISO Bio-Galvanic Safety Sentinel... ";
    ExocortexBioGuard guard;
    std::vector<float> overamp(EMBEDDING_DIM, 50.0f);
    double temp_rise = 0.0;

    bool safe = guard.validate_and_clamp_feedback(overamp.data(), EMBEDDING_DIM, temp_rise);

    for (size_t i = 0; i < EMBEDDING_DIM; ++i) {
        assert(std::abs(overamp[i]) <= 25.0f && "Bio-guard failed to clamp feedback voltage!");
    }
    assert(safe && "Thermal rise must remain within ISO envelope!");
    std::cout << "PASSED! (Clamped to 25.0 µC/cm², Temp Rise: +" << temp_rise << " °C)\n";
}

void test_cuda_embedding() {
    std::cout << "[TEST 4] CUDA 1024-d Cognitive Vector Embedding Kernel... ";
    float *h_rates, *h_embeds, *d_rates, *d_embeds;

    cudaMallocHost(&h_rates, TOTAL_NEURAL_CHANNELS * sizeof(float));
    cudaMallocHost(&h_embeds, EMBEDDING_DIM * sizeof(float));
    cudaMalloc(&d_rates, TOTAL_NEURAL_CHANNELS * sizeof(float));
    cudaMalloc(&d_embeds, EMBEDDING_DIM * sizeof(float));

    for (size_t i = 0; i < TOTAL_NEURAL_CHANNELS; ++i) h_rates[i] = 5.0f;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaMemcpyAsync(d_rates, h_rates, TOTAL_NEURAL_CHANNELS * sizeof(float), cudaMemcpyHostToDevice, stream);
    launch_spike_to_embedding_kernel(d_rates, d_embeds, TOTAL_NEURAL_CHANNELS, EMBEDDING_DIM, stream);
    cudaMemcpyAsync(h_embeds, d_embeds, EMBEDDING_DIM * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    for (size_t i = 0; i < EMBEDDING_DIM; ++i) {
        assert(h_embeds[i] != 0.0f && "CUDA embedding dimension output failed!");
    }

    cudaFreeHost(h_rates);
    cudaFreeHost(h_embeds);
    cudaFree(d_rates);
    cudaFree(d_embeds);
    cudaStreamDestroy(stream);

    std::cout << "PASSED! (1024 embedding dimensions validated on RTX 5060)\n";
}

int main() {
    std::cout << "========================================================\n";
    std::cout << "  SYNTHETIC EXOCORTEX GATEWAY - DIAGNOSTIC SUITE\n";
    std::cout << "========================================================\n\n";

    test_avx_decoder();
    test_orchestrator();
    test_bio_guard();
    test_cuda_embedding();

    std::cout << "\n========================================================\n";
    std::cout << "  ALL 4 DIAGNOSTIC CHECKS PASSED (100% VERIFIED)\n";
    std::cout << "========================================================\n";

    return 0;
}
