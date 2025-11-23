import matplotlib.pyplot as plt
import numpy as np
import json
import os

def compute_metrics(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    num_runs = data["n_runs"]
    num_pairs = data["num_pairs"]

    metrics = {
        "round_trip_latencies": [0] * num_pairs,
        "round_trip_throughputs": [0] * num_pairs,
        "single_trip_latencies_client": [0] * num_pairs,
        "single_trip_latencies_server": [0] * num_pairs,
        "fabric_latency_client": [0] * num_pairs,
        "fabric_latency_server": [0] * num_pairs,
    }

    for run_idx in range(num_runs):
        for pair_idx in range(num_pairs):
            entry = data[f"run{run_idx}"][f"pair{pair_idx}"]

            round_trip_latency = entry["client_recv_end"] - entry["client_trans_start"]
            round_trip_throughput = data["message_size"] / (round_trip_latency / 1e9) # bytes per second (ns -> s)
            single_trip_latency_client = entry["server_recv_end"] - entry["client_trans_start"]
            single_trip_latency_server = entry["client_recv_end"] - entry["server_trans_start"]
            fabric_latency_client = entry["client_recv_end"] - entry["client_recv_start"]
            fabric_latency_server = entry["server_recv_end"] - entry["server_recv_start"]

            metrics["round_trip_latencies"][pair_idx] += round_trip_latency / num_runs
            metrics["round_trip_throughputs"][pair_idx] += round_trip_throughput / num_runs
            metrics["single_trip_latencies_client"][pair_idx] += single_trip_latency_client / num_runs
            metrics["single_trip_latencies_server"][pair_idx] += single_trip_latency_server / num_runs
            metrics["fabric_latency_client"][pair_idx] += fabric_latency_client / num_runs
            metrics["fabric_latency_server"][pair_idx] += fabric_latency_server / num_runs

    return metrics

if __name__ == "__main__":
    data_dir = "data/"
    categories = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    all_metrics = {}
    os.makedirs('plots', exist_ok=True)
    for category in categories:
        category_path = os.path.join(data_dir, category)
        json_files = [f for f in os.listdir(category_path) if f.endswith('.json')]
        all_metrics[category] = []
        for json_file in json_files:
            json_path = os.path.join(category_path, json_file)
            metrics = compute_metrics(json_path)
            all_metrics[category].append((json_file, metrics))

    # create four plots in a 2x2 grid
    # 1. Round-trip Latency vs Message Size (a line for each category and pair)
    # 2. Round-trip Throughput vs Message Size (a line for each category and pair)
    # 3. Single-trip Latency (Client) vs Message Size (a line for each category, pair, and server/client)
    # 4. Fabric Latency (Client) vs Message Size (a line for each category, pair, and server/client)
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Cross-Warp Benchmark Results')
    
    hwd_alias = {
        "NVIDIA_GeForce_RTX_5090": "RTX 5090",
    }

    for category, metrics_list in all_metrics.items():
        for json_file, metrics in metrics_list:
            msg_size = int(json_file.split('_')[1].replace('B', ''))
            num_pairs = len(metrics["round_trip_latencies"])
            x = [msg_size] * num_pairs
            category_label = hwd_alias.get(category, category) + f" ({msg_size}B)"

            # Round-trip Latency
            axs[0, 0].plot(x, metrics["round_trip_latencies"], marker='o', label=f"{category_label}")

            # Round-trip Throughput
            axs[0, 1].plot(x, metrics["round_trip_throughputs"], marker='o', label=f"{category_label}")

            # Single-trip Latency (Client)
            axs[1, 0].plot(x, metrics["single_trip_latencies_client"], marker='o', label=f"{category_label}")

            # Fabric Latency (Client)
            axs[1, 1].plot(x, metrics["fabric_latency_client"], marker='o', label=f"{category_label}")

    axs[0, 0].set_xscale('log', base=2)
    axs[0, 0].set_yscale('log')
    axs[0, 0].set_title('Round-trip Latency vs Message Size')
    axs[0, 0].set_xlabel('Message Size (bytes)')
    axs[0, 0].set_ylabel('Round-trip Latency (ns)')
    axs[0, 0].legend()
    
    axs[0, 1].set_xscale('log', base=2)
    axs[0, 1].set_yscale('log')
    axs[0, 1].set_title('Round-trip Throughput vs Message Size')
    axs[0, 1].set_xlabel('Message Size (bytes)')
    axs[0, 1].set_ylabel('Round-trip Throughput (bytes/s)')
    axs[0, 1].legend()

    axs[1, 0].set_xscale('log', base=2)
    axs[1, 0].set_yscale('log')
    axs[1, 0].set_title('Single-trip Latency (Client) vs Message Size')
    axs[1, 0].set_xlabel('Message Size (bytes)')
    axs[1, 0].set_ylabel('Single-trip Latency (ns)')
    axs[1, 0].legend()

    axs[1, 1].set_xscale('log', base=2)
    axs[1, 1].set_yscale('log')
    axs[1, 1].set_title('Fabric Latency (Client) vs Message Size')
    axs[1, 1].set_xlabel('Message Size (bytes)')
    axs[1, 1].set_ylabel('Fabric Latency (ns)')
    axs[1, 1].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('plots/cross_warp_benchmark_results.png')