#ifndef AVX_SPIKE_DENSITY_DECODER_HPP
#define AVX_SPIKE_DENSITY_DECODER_HPP

#include <immintrin.h>
#include <vector>
#include <cmath>

constexpr size_t TOTAL_NEURAL_CHANNELS = 65536;
constexpr size_t EMBEDDING_DIM = 1024;

class AVXSpikeDensityDecoder {
public:
    AVXSpikeDensityDecoder() {
        weights_.resize(TOTAL_NEURAL_CHANNELS);
        for (size_t i = 0; i < TOTAL_NEURAL_CHANNELS; ++i) {
            weights_[i] = 0.05f + 0.95f * std::exp(-static_cast<float>(i % 128) / 32.0f);
        }
    }

    void process_spike_rates(const float* __restrict raw_spikes, float* __restrict rate_density) {
        __m256 v_scale = _mm256_set1_ps(1.2f);
        __m256 v_zero = _mm256_setzero_ps();

        for (size_t i = 0; i < TOTAL_NEURAL_CHANNELS; i += 8) {
            __m256 v_spk = _mm256_loadu_ps(&raw_spikes[i]);
            __m256 v_w = _mm256_loadu_ps(&weights_[i]);

            __m256 v_rate = _mm256_mul_ps(v_spk, v_w);
            __m256 v_scaled = _mm256_fmadd_ps(v_rate, v_scale, v_zero);
            __m256 v_relu = _mm256_max_ps(v_zero, v_scaled);

            _mm256_storeu_ps(&rate_density[i], v_relu);
        }
    }

private:
    std::vector<float> weights_;
};

#endif // AVX_SPIKE_DENSITY_DECODER_HPP
