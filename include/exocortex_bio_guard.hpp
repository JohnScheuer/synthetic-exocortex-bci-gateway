#ifndef EXOCORTEX_BIO_GUARD_HPP
#define EXOCORTEX_BIO_GUARD_HPP

#include <vector>
#include <cmath>

class ExocortexBioGuard {
public:
    static constexpr double MAX_TISSUE_TEMP_RISE_C = 0.50;
    static constexpr float MAX_CHARGE_DENSITY_UC = 25.0f;

    ExocortexBioGuard() : window_idx_(0), accumulated_heat_(0.0) {
        energy_window_.resize(100, 0.0);
    }

    bool validate_and_clamp_feedback(float* feedback_vectors, size_t dim, double& temp_rise) {
        double cycle_energy = 0.0;
        for (size_t i = 0; i < dim; ++i) {
            if (std::abs(feedback_vectors[i]) > MAX_CHARGE_DENSITY_UC) {
                feedback_vectors[i] = (feedback_vectors[i] > 0) ? MAX_CHARGE_DENSITY_UC : -MAX_CHARGE_DENSITY_UC;
            }
            cycle_energy += feedback_vectors[i] * feedback_vectors[i] * 0.00001;
        }

        accumulated_heat_ -= energy_window_[window_idx_];
        energy_window_[window_idx_] = cycle_energy;
        accumulated_heat_ += cycle_energy;
        window_idx_ = (window_idx_ + 1) % 100;

        temp_rise = (accumulated_heat_ / 100.0) * 0.003;
        return (temp_rise < MAX_TISSUE_TEMP_RISE_C);
    }

private:
    size_t window_idx_;
    double accumulated_heat_;
    std::vector<double> energy_window_;
};

#endif // EXOCORTEX_BIO_GUARD_HPP
