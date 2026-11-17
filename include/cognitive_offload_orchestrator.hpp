#ifndef COGNITIVE_OFFLOAD_ORCHESTRATOR_HPP
#define COGNITIVE_OFFLOAD_ORCHESTRATOR_HPP

#include <vector>
#include <numeric>
#include <cmath>

enum class CognitiveState {
    LOCAL_PROCESSING,
    GPU_EXOCORTEX_OFFLOAD
};

class CognitiveOffloadOrchestrator {
public:
    CognitiveOffloadOrchestrator() : entropy_threshold_(42.5f) {}

    CognitiveState evaluate_cognitive_load(const float* rate_density, float& current_entropy) {
        float sample_sum = 0.0f;
        for (size_t i = 0; i < 1024; ++i) {
            sample_sum += rate_density[i];
        }
        current_entropy = sample_sum / 1024.0f;

        if (current_entropy > entropy_threshold_) {
            return CognitiveState::GPU_EXOCORTEX_OFFLOAD;
        }
        return CognitiveState::LOCAL_PROCESSING;
    }

private:
    float entropy_threshold_;
};

#endif // COGNITIVE_OFFLOAD_ORCHESTRATOR_HPP
