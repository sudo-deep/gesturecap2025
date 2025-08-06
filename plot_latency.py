import matplotlib.pyplot as plt
import numpy as np

# log_file = "latency_logs/delay_delta5_freq200_th11_aux.txt"
# log_file = "latency_logs/delay_delta5_freq200_th60_aux.txt"
# log_file = "latency_logs/delay_raw_freq200_th200_aux.txt"
# log_file = "freezed_log_A15/delay_raw_freq200_th80_aux.txt"
log_file = "latency_logs/delay_test_freqtest_thtest_test.txt"

# Read and extract latency values
latencies = []
with open(log_file, 'r') as f:
    for line in f:
        try:
            parts = line.strip().split(',')
            if len(parts) == 2:
                latency_str = parts[1].strip().split()[0]
                latency = int(latency_str)
                latencies.append(latency)
        except:
            continue

if not latencies:
    print("No latency data found.")
    exit()

# Convert to numpy array
latencies = np.array(latencies)

# Compute mean and std of full data
mean_all = np.mean(latencies)
std_all = np.std(latencies)

# Filter out outliers beyond 3 standard deviations
z_scores = (latencies - mean_all) / std_all
filtered_latencies = latencies[np.abs(z_scores) < 3]

# Compute statistics after filtering
mean_filtered = np.mean(filtered_latencies)
std_filtered = np.std(filtered_latencies)
min_val = np.min(filtered_latencies)
max_val = np.max(filtered_latencies)
median_val = np.median(filtered_latencies)

# Print stats
print(f"Original samples: {len(latencies)}")
print(f"Filtered samples: {len(filtered_latencies)}")
print(f"Min latency (filtered): {min_val} ms")
print(f"Max latency (filtered): {max_val} ms")
print(f"Median latency (filtered): {median_val} ms")
print(f"Mean latency (filtered): {mean_filtered:.2f} ms")
print(f"Standard deviation (filtered): {std_filtered:.2f} ms")

# Plot
plt.figure(figsize=(10, 5))
plt.plot(filtered_latencies, marker='o', linestyle='-', color='blue')
plt.title("Button to Audio Latency (Outliers Removed)")
plt.xlabel("Sample Index")
plt.ylabel("Latency (ms)")
plt.grid(True)

# Stats lines
plt.axhline(mean_filtered, color='green', linestyle='--', label=f'Mean = {mean_filtered:.2f} ms')
plt.axhline(mean_filtered + std_filtered, color='orange', linestyle='--', label=f'+1σ = {mean_filtered + std_filtered:.2f} ms')
plt.axhline(mean_filtered - std_filtered, color='orange', linestyle='--', label=f'-1σ = {mean_filtered - std_filtered:.2f} ms')
plt.axhline(median_val, color='purple', linestyle=':', label=f'Median = {median_val:.2f} ms')
plt.axhline(min_val, color='gray', linestyle='--', label=f'Min = {min_val} ms')
plt.axhline(max_val, color='gray', linestyle='--', label=f'Max = {max_val} ms')

plt.legend()
plt.tight_layout()
plt.show()
