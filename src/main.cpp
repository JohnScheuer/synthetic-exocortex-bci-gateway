#include "avx2_signal_decoder.hpp"
#include "cognitive_orchestrator.hpp"
#include "iso_bio_guard.hpp"
#include "cuda_embedding_engine.cuh"
#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <numeric>
#include <algorithm>
#include <cuda_runtime.h>

int main() {
    cudaSetDeviceFlags(cudaDeviceScheduleSpin);

    std::cout << "========================================================\n";
    std::cout << "  SYNTHETIC EXOCORTEX BCI GATEWAY - BENCHMARK ENGINE\n";
    std::cout << "  Author: Joao Felipe de Souza (github.com/JohnScheuer)\n";
    std::cout << "========================================================\n\n";

    std::cout << "[GPU Target] NVIDIA GeForce RTX 5060 (Compute 12.0)\n";
    std::cout << "[Pipeline] Executing 10,000 Cycles over 65,536 Neural Channels @ 1000 Hz...\n\n";

    AVX2SignalDecoder decoder;
    CognitiveOrchestrator orchestrator(42.5f);
    ISOBioGuard bio_guard(0.50f, 30.0f);
    CUDAEmbeddingEngine cuda_engine;

    std::vector<float> raw_signals(CHANNELS * SPIKE_WINDOW, 0.5f);
    std::vector<float> spike_rates(CHANNELS, 0.0f);
    std::vector<float> embeddings(EMBEDDING_DIM, 0.0f);

    // CUDA Context Warmup
    cuda_engine.launch_embedding_kernel(spike_rates.data(), embeddings.data());

    const int TOTAL_CYCLES = 10000;
    std::vector<double> total_latencies(TOTAL_CYCLES, 0.0);
    std::vector<double> avx2_latencies(TOTAL_CYCLES, 0.0);
    std::vector<double> cuda_latencies(TOTAL_CYCLES, 0.0);
    std::vector<double> offload_latencies(TOTAL_CYCLES, 0.0);
    std::vector<double> bioguard_latencies(TOTAL_CYCLES, 0.0);
    std::vector<float> temp_rises(TOTAL_CYCLES, 0.0f);
    std::vector<float> entropies(TOTAL_CYCLES, 0.0f);

    for (int i = 0; i < TOTAL_CYCLES; ++i) {
        auto t_start = std::chrono::steady_clock::now();

        // Stage 1: AVX2 Decode
        auto t0 = std::chrono::steady_clock::now();
        decoder.decode_spike_rates(raw_signals.data(), spike_rates.data());
        auto t1 = std::chrono::steady_clock::now();
        avx2_latencies[i] = std::chrono::duration<double, std::micro>(t1 - t0).count();

        // Stage 2: CUDA Embedding
        cuda_engine.launch_embedding_kernel(spike_rates.data(), embeddings.data());
        auto t2 = std::chrono::steady_clock::now();
        cuda_latencies[i] = std::chrono::duration<double, std::micro>(t2 - t1).count();

        // Stage 3: Orchestrator
        float current_entropy = 0.0f;
        orchestrator.evaluate_cognitive_load(embeddings.data(), current_entropy);
        entropies[i] = current_entropy;
        auto t3 = std::chrono::steady_clock::now();
        offload_latencies[i] = std::chrono::duration<double, std::micro>(t3 - t2).count();

        // Stage 4: BioGuard Safety
        float current_temp_rise = 0.0f;
        bio_guard.validate_and_clamp_feedback(embeddings.data(), EMBEDDING_DIM, current_temp_rise);
        temp_rises[i] = current_temp_rise;
        auto t4 = std::chrono::steady_clock::now();
        bioguard_latencies[i] = std::chrono::duration<double, std::micro>(t4 - t3).count();

        total_latencies[i] = std::chrono::duration<double, std::micro>(t4 - t_start).count();
    }

    std::ofstream csv("benchmark_telemetry.csv");
    if (csv.is_open()) {
        csv << "cycle_id,total_latency_us,avx2_decode_us,cuda_embed_us,offload_us,bioguard_us,temp_rise_c,entropy_val\n";
        for (int i = 0; i < TOTAL_CYCLES; ++i) {
            csv << i << ","
                << total_latencies[i] << ","
                << avx2_latencies[i] << ","
                << cuda_latencies[i] << ","
                << offload_latencies[i] << ","
                << bioguard_latencies[i] << ","
                << temp_rises[i] << ","
                << entropies[i] << "\n";
        }
        csv.close();
        std::cout << "[Telemetry] Wrote " << TOTAL_CYCLES << " cycles to benchmark_telemetry.csv\n";
    }

    std::vector<double> sorted_latencies = total_latencies;
    std::sort(sorted_latencies.begin(), sorted_latencies.end());

    double min_lat = sorted_latencies[0];
    double mean_lat = std::accumulate(sorted_latencies.begin(), sorted_latencies.end(), 0.0) / TOTAL_CYCLES;
    double p50_lat = sorted_latencies[static_cast<size_t>(TOTAL_CYCLES * 0.50)];
    double p90_lat = sorted_latencies[static_cast<size_t>(TOTAL_CYCLES * 0.90)];
    double p99_lat = sorted_latencies[static_cast<size_t>(TOTAL_CYCLES * 0.99)];
    double p999_lat = sorted_latencies[static_cast<size_t>(TOTAL_CYCLES * 0.999)];
    double max_lat = sorted_latencies[TOTAL_CYCLES - 1];

    int violations = 0;
    for (double lat : total_latencies) {
        if (lat > 500.0) violations++;
    }
    double violation_rate = (double)violations / TOTAL_CYCLES * 100.0;
    float max_temp = *std::max_element(temp_rises.begin(), temp_rises.end());

    std::cout << "\n========================================================\n";
    std::cout << "  EMPIRICAL LATENCY BENCHMARK RESULTS (RTX 5060 + Ryzen 5600X)\n";
    std::cout << "========================================================\n";
    std::cout << "  Min Latency        : " << min_lat << " us\n";
    std::cout << "  Mean Latency       : " << mean_lat << " us\n";
    std::cout << "  P50 Latency (Median): " << p50_lat << " us\n";
    std::cout << "  P90 Latency        : " << p90_lat << " us\n";
    std::cout << "  P99 Latency (SLA)  : " << p99_lat << " us (Target: < 500.00 us)\n";
    std::cout << "  P99.9 Latency      : " << p999_lat << " us\n";
    std::cout << "  Max Latency        : " << max_lat << " us\n";
    std::cout << "  SLA Violation Rate : " << violation_rate << " %\n";
    std::cout << "  Max Tissue Heating : +" << max_temp << " deg C (ISO Limit: +0.50 deg C)\n";
    std::cout << "========================================================\n";

    return 0;
}
