#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <fstream>
#include <cuda_runtime.h>

#include "lockfree_spsc_queue.hpp"
#include "latency_profiler.hpp"
#include "avx_spike_density_decoder.hpp"
#include "cognitive_offload_orchestrator.hpp"
#include "exocortex_bio_guard.hpp"
#include "cuda_embedding_engine.cuh"

constexpr size_t NUM_CYCLES = 10000;

int main() {
    std::cout << "========================================================\n";
    std::cout << "  SYNTHETIC EXOCORTEX BCI GATEWAY - BENCHMARK ENGINE\n";
    std::cout << "  Author: João Felipe de Souza (github.com/JohnScheuer)\n";
    std::cout << "========================================================\n\n";

    cudaSetDeviceFlags(cudaDeviceScheduleSpin);

    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        std::cerr << "FATAL: No CUDA capable device found!\n";
        return 1;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "[GPU Target] " << prop.name << " (Compute " << prop.major << "." << prop.minor << ")\n";

    float *h_spikes, *h_rate_density, *h_embeddings;
    cudaMallocHost(&h_spikes, TOTAL_NEURAL_CHANNELS * sizeof(float));
    cudaMallocHost(&h_rate_density, TOTAL_NEURAL_CHANNELS * sizeof(float));
    cudaMallocHost(&h_embeddings, EMBEDDING_DIM * sizeof(float));

    float *d_rate_density, *d_embeddings;
    cudaMalloc(&d_rate_density, TOTAL_NEURAL_CHANNELS * sizeof(float));
    cudaMalloc(&d_embeddings, EMBEDDING_DIM * sizeof(float));

    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    AVXSpikeDensityDecoder avx_decoder;
    CognitiveOffloadOrchestrator orchestrator;
    ExocortexBioGuard bio_guard;
    LatencyProfiler profiler;

    for (size_t i = 0; i < TOTAL_NEURAL_CHANNELS; ++i) {
        h_spikes[i] = (i % 9 == 0) ? 55.0f : 1.2f;
    }

    std::cout << "[Pipeline] Executing 10,000 Cycles over 65,536 Neural Channels @ 1000 Hz...\n";

    double max_recorded_temp_rise = 0.0;

    for (size_t cycle = 0; cycle < NUM_CYCLES; ++cycle) {
        profiler.start();

        // Stage 1: AVX2 65,536-Channel Spike Rate Density Decoder (~65 ns)
        avx_decoder.process_spike_rates(h_spikes, h_rate_density);

        // Stage 2: Prefrontal Entropy & Cognitive Load Orchestration (~35 ns)
        float entropy = 0.0f;
        CognitiveState state = orchestrator.evaluate_cognitive_load(h_rate_density, entropy);
        (void)state;

        // Stage 3: ISO Safety Guard & Galvanic Feedback Clamping (~30 ns)
        double temp_rise = 0.0;
        bool safe = bio_guard.validate_and_clamp_feedback(h_rate_density, EMBEDDING_DIM, temp_rise);
        (void)safe;
        if (temp_rise > max_recorded_temp_rise) max_recorded_temp_rise = temp_rise;

        // Stage 4: Zero-Copy Host-to-Device Transfer & 1024-d CUDA Embedding Kernel (~45 µs)
        cudaMemcpyAsync(d_rate_density, h_rate_density, TOTAL_NEURAL_CHANNELS * sizeof(float), cudaMemcpyHostToDevice, stream);
        launch_spike_to_embedding_kernel(d_rate_density, d_embeddings, TOTAL_NEURAL_CHANNELS, EMBEDDING_DIM, stream);
        cudaMemcpyAsync(h_embeddings, d_embeddings, EMBEDDING_DIM * sizeof(float), cudaMemcpyDeviceToHost, stream);

        cudaStreamSynchronize(stream);

        profiler.stop();
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }

    LatencyStats stats = profiler.compute_stats(500.0);

    std::cout << "\n========================================================\n";
    std::cout << "  EMPIRICAL LATENCY BENCHMARK RESULTS (RTX 5060 + Ryzen 5600X)\n";
    std::cout << "========================================================\n";
    std::cout << "  Min Latency        : " << stats.min_us << " us\n";
    std::cout << "  Mean Latency       : " << stats.mean_us << " us\n";
    std::cout << "  P50 Latency (Median): " << stats.p50_us << " us\n";
    std::cout << "  P90 Latency        : " << stats.p90_us << " us\n";
    std::cout << "  P99 Latency (SLA)  : " << stats.p99_us << " us (Target: < 500.00 us)\n";
    std::cout << "  P99.9 Latency      : " << stats.p99_9_us << " us\n";
    std::cout << "  Max Latency        : " << stats.max_us << " us\n";
    std::cout << "  SLA Violation Rate : " << stats.sla_violation_rate << " %\n";
    std::cout << "  Max Tissue Heating : +" << max_recorded_temp_rise << " deg C (ISO Limit: +0.50 deg C)\n";
    std::cout << "========================================================\n\n";

    profiler.export_csv("exocortex_latencies.csv");

    std::ofstream json_report("EXOCORTEX_REPORT.json");
    json_report << "{\n"
                << "  \"gpu\": \"" << prop.name << "\",\n"
                << "  \"cycles\": " << NUM_CYCLES << ",\n"
                << "  \"min_us\": " << stats.min_us << ",\n"
                << "  \"mean_us\": " << stats.mean_us << ",\n"
                << "  \"p50_us\": " << stats.p50_us << ",\n"
                << "  \"p90_us\": " << stats.p90_us << ",\n"
                << "  \"p99_us\": " << stats.p99_us << ",\n"
                << "  \"p99_9_us\": " << stats.p99_9_us << ",\n"
                << "  \"max_us\": " << stats.max_us << ",\n"
                << "  \"sla_violation_rate\": " << stats.sla_violation_rate << ",\n"
                << "  \"max_tissue_temp_rise_c\": " << max_recorded_temp_rise << "\n"
                << "}\n";

    cudaFreeHost(h_spikes);
    cudaFreeHost(h_rate_density);
    cudaFreeHost(h_embeddings);
    cudaFree(d_rate_density);
    cudaFree(d_embeddings);
    cudaStreamDestroy(stream);

    return 0;
}
