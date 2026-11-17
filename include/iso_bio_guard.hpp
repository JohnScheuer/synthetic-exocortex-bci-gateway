#ifndef ISO_BIO_GUARD_HPP
#define ISO_BIO_GUARD_HPP

#include <vector>
#include <algorithm>
#include <cmath>

class ISOBioGuard {
private:
    float temp_limit_c_;
    float charge_density_limit_uc_;
    float accumulated_temp_rise_;

public:
    explicit ISOBioGuard(float max_temp = 0.50f, float max_charge = 30.0f)
        : temp_limit_c_(max_temp), charge_density_limit_uc_(max_charge), accumulated_temp_rise_(0.0f) {}

    inline bool validate_and_clamp_feedback(float* feedback_vector, int size, float& current_temp_rise) {
        bool clamped = false;
        float total_energy = 0.0f;

        for (int i = 0; i < size; ++i) {
            if (feedback_vector[i] > charge_density_limit_uc_) {
                feedback_vector[i] = charge_density_limit_uc_;
                clamped = true;
            }
            total_energy += feedback_vector[i] * feedback_vector[i];
        }

        accumulated_temp_rise_ += (total_energy * 1e-8f) - (accumulated_temp_rise_ * 0.01f);
        if (accumulated_temp_rise_ > temp_limit_c_) {
            accumulated_temp_rise_ = temp_limit_c_;
        }

        current_temp_rise = accumulated_temp_rise_;
        return !clamped && (accumulated_temp_rise_ < temp_limit_c_);
    }
};

#endif
