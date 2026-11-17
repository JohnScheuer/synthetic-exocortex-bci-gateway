#ifndef LATENCY_PROFILER_HPP
#define LATENCY_PROFILER_HPP

#include <chrono>
#include <vector>
#include <numeric>
#include <algorithm>
#include <fstream>

struct LatencyStats {
    double min_us;
    double mean_us;
    double p50_us;
    double p90_us;
    double p99_us;
    double p99_9_us;
    double max_us;
    double sla_violation_rate;
};

class LatencyProfiler {
public:
    using Clock = std::chrono::high_resolution_clock;

    void start() { start_time_ = Clock::now(); }

    double stop() {
        auto end_time = Clock::now();
        double duration_us = std::chrono::duration<double, std::micro>(end_time - start_time_).count();
        samples_.push_back(duration_us);
        return duration_us;
    }

    LatencyStats compute_stats(double sla_threshold_us = 500.0) const {
        if (samples_.empty()) return {};

        std::vector<double> sorted = samples_;
        std::sort(sorted.begin(), sorted.end());

        size_t n = sorted.size();
        double sum = std::accumulate(sorted.begin(), sorted.end(), 0.0);

        LatencyStats stats;
        stats.min_us = sorted.front();
        stats.max_us = sorted.back();
        stats.mean_us = sum / n;
        stats.p50_us = sorted[static_cast<size_t>(n * 0.50)];
        stats.p90_us = sorted[static_cast<size_t>(n * 0.90)];
        stats.p99_us = sorted[static_cast<size_t>(n * 0.99)];
        stats.p99_9_us = sorted[static_cast<size_t>(n * 0.999)];

        size_t violations = 0;
        for (double s : sorted) {
            if (s > sla_threshold_us) violations++;
        }
        stats.sla_violation_rate = (static_cast<double>(violations) / n) * 100.0;

        return stats;
    }

    void export_csv(const std::string& filename) const {
        std::ofstream file(filename);
        file << "cycle,latency_us\n";
        for (size_t i = 0; i < samples_.size(); ++i) {
            file << i << "," << samples_[i] << "\n";
        }
    }

private:
    Clock::time_point start_time_;
    std::vector<double> samples_;
};

#endif // LATENCY_PROFILER_HPP
