#ifndef AVX2_SIGNAL_DECODER_HPP
#define AVX2_SIGNAL_DECODER_HPP

#include <immintrin.h>
#include <vector>
#include <cmath>

constexpr int CHANNELS = 65536;
constexpr int SPIKE_WINDOW = 32;

class AVX2SignalDecoder {
public:
    inline void decode_spike_rates(const float* raw_signals, float* spike_rates) {
        for (int i = 0; i < CHANNELS; i += 8) {
            __m256 sum = _mm256_setzero_ps();
            for (int w = 0; w < SPIKE_WINDOW; ++w) {
                __m256 sig = _mm256_loadu_ps(&raw_signals[i + w * CHANNELS]);
                sum = _mm256_add_ps(sum, sig);
            }
            __m256 scale = _mm256_set1_ps(1000.0f / static_cast<float>(SPIKE_WINDOW));
            __m256 rates = _mm256_mul_ps(sum, scale);
            _mm256_storeu_ps(&spike_rates[i], rates);
        }
    }
};

#endif
