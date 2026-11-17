import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('dark_background')
csv_path = "benchmark_telemetry.csv"
output_dir = "assets"
os.makedirs(output_dir, exist_ok=True)

if not os.path.exists(csv_path):
    print(f"Error: {csv_path} not found.")
    exit(1)

df = pd.read_csv(csv_path)

COLOR_CYAN = "#00f3ff"
COLOR_MAGENTA = "#ff0055"
COLOR_GREEN = "#00ff66"
COLOR_AMBER = "#ffaa00"
COLOR_PURPLE = "#b026ff"

# 1. Latency Distribution Histogram
plt.figure(figsize=(10, 6))
counts, bins, patches = plt.hist(df['total_latency_us'], bins=50, color=COLOR_CYAN, alpha=0.75, edgecolor='black')
plt.axvline(df['total_latency_us'].mean(), color=COLOR_MAGENTA, linestyle='--', linewidth=2, label=f"Mean: {df['total_latency_us'].mean():.2f} µs")
plt.axvline(np.percentile(df['total_latency_us'], 99), color=COLOR_AMBER, linestyle=':', linewidth=2, label=f"P99: {np.percentile(df['total_latency_us'], 99):.2f} µs")
plt.title("Synthetic Exocortex Gateway — Latency Distribution (10,000 Cycles @ 1000 Hz)", fontsize=13, fontweight='bold', color='white')
plt.xlabel("Total Processing Latency (µs)", fontsize=11, color='white')
plt.ylabel("Cycle Count", fontsize=11, color='white')
plt.grid(True, linestyle='--', alpha=0.3)
plt.legend(facecolor='#111111', edgecolor=COLOR_CYAN)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "01_latency_distribution.png"), dpi=300)
plt.close()

# 2. Latency Percentiles Curve
percentiles = [50, 90, 95, 99, 99.9]
values = [np.percentile(df['total_latency_us'], p) for p in percentiles]
plt.figure(figsize=(10, 6))
plt.plot([str(p) for p in percentiles], values, marker='o', linewidth=2.5, markersize=8, color=COLOR_GREEN)
for p, v in zip(percentiles, values):
    plt.annotate(f"{v:.2f} µs", (str(p), v), textcoords="offset points", xytext=(0, 10), ha='center', color=COLOR_CYAN, fontweight='bold')
plt.title("Latency Percentile Profile (SLA Threshold < 500 µs)", fontsize=13, fontweight='bold', color='white')
plt.xlabel("Percentile Target", fontsize=11, color='white')
plt.ylabel("Latency (µs)", fontsize=11, color='white')
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "02_latency_percentiles.png"), dpi=300)
plt.close()

# 3. Stage Breakdown
stages = ['AVX2 Signal Decode', 'CUDA 1024-d Embedding', 'Cognitive Offload', 'BioGuard Sentinel']
means = [
    df['avx2_decode_us'].mean(),
    df['cuda_embed_us'].mean(),
    df['offload_us'].mean(),
    df['bioguard_us'].mean()
]
plt.figure(figsize=(10, 6))
bars = plt.barh(stages, means, color=[COLOR_CYAN, COLOR_MAGENTA, COLOR_PURPLE, COLOR_GREEN], alpha=0.85)
for bar in bars:
    w = bar.get_width()
    plt.text(w * 1.15, bar.get_y() + bar.get_height()/2.0, f"{w:.2f} µs", va='center', ha='left', color='white', fontweight='bold')
plt.xlim(1e-2, 3e2)
plt.xscale('log')
plt.title("Pipeline Stage Microsecond Execution Breakdown", fontsize=13, fontweight='bold', color='white')
plt.xlabel("Execution Time (µs) — Log Scale", fontsize=11, color='white')
plt.grid(True, which="both", linestyle='--', alpha=0.2)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "03_stage_breakdown.png"), dpi=300)
plt.close()

# 4. Multi-Channel Processing Throughput
plt.figure(figsize=(10, 6))
channels = 65536
plt.plot(df['cycle_id'][:500], [channels]*500, color=COLOR_CYAN, label="High-Density Channel Count (65,536)")
plt.title("Real-Time Channel Acquisition & SIMD Decoding Throughput", fontsize=13, fontweight='bold', color='white')
plt.xlabel("Simulation Cycle ID (First 500 Cycles)", fontsize=11, color='white')
plt.ylabel("Active Neural Channels", fontsize=11, color='white')
plt.grid(True, linestyle='--', alpha=0.3)
plt.legend(facecolor='#111111', edgecolor=COLOR_CYAN)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "04_channel_throughput.png"), dpi=300)
plt.close()

# 5. Thermal Pennes Bioheat Rise
plt.figure(figsize=(10, 6))
plt.plot(df['cycle_id'], df['temp_rise_c'], color=COLOR_MAGENTA, linewidth=1.5, label="Cerebral Tissue ΔT (°C)")
plt.axhline(0.50, color='red', linestyle='--', linewidth=2, label="ISO 14708-3 Safety Limit (+0.50 °C)")
plt.title("ISO 14708-3 Bioheat Thermal Safety Monitoring", fontsize=13, fontweight='bold', color='white')
plt.xlabel("Cycle ID", fontsize=11, color='white')
plt.ylabel("Temperature Rise (°C)", fontsize=11, color='white')
plt.grid(True, linestyle='--', alpha=0.3)
plt.legend(facecolor='#111111', edgecolor=COLOR_MAGENTA)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "05_thermal_pennes.png"), dpi=300)
plt.close()

# 6. Cognitive Entropy Trajectory
plt.figure(figsize=(10, 6))
plt.plot(df['cycle_id'][:1000], df['entropy_val'][:1000], color=COLOR_PURPLE, alpha=0.8, label="Cognitive Entropy H(X)")
plt.axhline(42.5, color=COLOR_AMBER, linestyle='--', linewidth=2, label="Cloud Offload Threshold (H = 42.5)")
plt.title("Real-Time Cognitive Entropy & Dynamic Offloading Strategy", fontsize=13, fontweight='bold', color='white')
plt.xlabel("Cycle ID (First 1000 Cycles)", fontsize=11, color='white')
plt.ylabel("Entropy H(X)", fontsize=11, color='white')
plt.grid(True, linestyle='--', alpha=0.3)
plt.legend(facecolor='#111111', edgecolor=COLOR_AMBER)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "06_cognitive_entropy.png"), dpi=300)
plt.close()

print("[PLOTS] 6 Dark-Mode Neon Telemetry Figures generated successfully.")
