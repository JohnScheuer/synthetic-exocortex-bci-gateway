#ifndef COGNITIVE_ORCHESTRATOR_HPP
#define COGNITIVE_ORCHESTRATOR_HPP

#include <vector>
#include <cmath>

constexpr int EMBEDDING_DIM = 1024;

enum class CognitiveState {
    LocalProcessing,
    CloudGPUOffload
};

class CognitiveOrchestrator {
private:
    float entropy_threshold_;
public:
    explicit CognitiveOrchestrator(float threshold = 42.5f) : entropy_threshold_(threshold) {}

    inline CognitiveState evaluate_cognitive_load(const float* embedding_vector, float& out_entropy) {
        float sum = 0.0f;
        for (int i = 0; i < EMBEDDING_DIM; ++i) {
            sum += std::abs(embedding_vector[i]);
        }
        out_entropy = 0.0f;
        if (sum > 0.0f) {
            for (int i = 0; i < EMBEDDING_DIM; ++i) {
                float p = std::abs(embedding_vector[i]) / sum;
                if (p > 1e-7f) {
                    out_entropy -= p * std::log2(p);
                }
            }
        }
        out_entropy *= 100.0f;

        if (out_entropy > entropy_threshold_) {
            return CognitiveState::CloudGPUOffload;
        }
        return CognitiveState::LocalProcessing;
    }
};

#endif
