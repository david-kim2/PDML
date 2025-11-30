import matplotlib.pyplot as plt
import numpy as np
import argparse
import json
import os


def format_bytes(x, pos):
    if x >= 1 << 30:   return f"{int(x / (1 << 30))}GB"
    elif x >= 1 << 20: return f"{int(x / (1 << 20))}MB"
    elif x >= 1 << 10: return f"{int(x / (1 << 10))}KB"
    else:              return f"{int(x)}B"


def time_format(x, pos):
    if x >= 1e9:   return f"{int(x / 1e9)}s"
    elif x >= 1e6: return f"{int(x / 1e6)}ms"
    elif x >= 1e3: return f"{int(x / 1e3)}µs"
    else:          return f"{int(x)}ns"


def plot_comparison_graphs(metrics, pairs, ignore_client, ignore_server):
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Cross-Block Comparison Benchmark Results', fontsize=16)

    for device, device_metrics in metrics.items():
        msg_sizes                    = np.array([entry["msg_size"] for entry in device_metrics])
        num_pairs                    = np.array([entry["num_pairs"] for entry in device_metrics])
        round_trip_latencies         = np.array([entry["metrics"]["round_trip_latency_avg"] for entry in device_metrics])
        round_trip_throughputs       = np.array([entry["metrics"]["round_trip_throughput_avg"] for entry in device_metrics])
        single_trip_latencies_client = np.array([entry["metrics"]["single_trip_latency_client_avg"] for entry in device_metrics])
        single_trip_latencies_server = np.array([entry["metrics"]["single_trip_latency_server_avg"] for entry in device_metrics])
        fabric_latencies_client      = np.array([entry["metrics"]["fabric_latency_client_avg"] for entry in device_metrics])
        fabric_latencies_server      = np.array([entry["metrics"]["fabric_latency_server_avg"] for entry in device_metrics])

        round_trip_latencies_std         = np.array([entry["metrics"]["round_trip_latency_std"] for entry in device_metrics])
        round_trip_throughputs_std       = np.array([entry["metrics"]["round_trip_throughput_std"] for entry in device_metrics])
        single_trip_latencies_client_std = np.array([entry["metrics"]["single_trip_latency_client_std"] for entry in device_metrics])
        single_trip_latencies_server_std = np.array([entry["metrics"]["single_trip_latency_server_std"] for entry in device_metrics])
        fabric_latencies_client_std      = np.array([entry["metrics"]["fabric_latency_client_std"] for entry in device_metrics])
        fabric_latencies_server_std      = np.array([entry["metrics"]["fabric_latency_server_std"] for entry in device_metrics])

        for pair in pairs:
            mask = (num_pairs == pair)
            if not np.any(mask): continue

            msg_sizes_subset                    = msg_sizes[mask]
            round_trip_latencies_subset         = round_trip_latencies[mask]
            round_trip_throughputs_subset       = round_trip_throughputs[mask]
            single_trip_latencies_client_subset = single_trip_latencies_client[mask]
            single_trip_latencies_server_subset = single_trip_latencies_server[mask]
            fabric_latencies_client_subset      = fabric_latencies_client[mask]
            fabric_latencies_server_subset      = fabric_latencies_server[mask]

            round_trip_latencies_std_subset         = round_trip_latencies_std[mask]
            round_trip_throughputs_std_subset       = round_trip_throughputs_std[mask]
            single_trip_latencies_client_std_subset = single_trip_latencies_client_std[mask]
            single_trip_latencies_server_std_subset = single_trip_latencies_server_std[mask]
            fabric_latencies_client_std_subset      = fabric_latencies_client_std[mask]
            fabric_latencies_server_std_subset      = fabric_latencies_server_std[mask]

            category_label = device + f" (P={pair})"
            axs[0, 0].errorbar(msg_sizes_subset, round_trip_latencies_subset, yerr=round_trip_latencies_std_subset,
                                marker='o', label=f"{category_label}")
            axs[0, 1].errorbar(msg_sizes_subset, round_trip_throughputs_subset, yerr=round_trip_throughputs_std_subset,
                                marker='o', label=f"{category_label}")

            if not ignore_client:
                axs[1, 0].errorbar(msg_sizes_subset, single_trip_latencies_client_subset, yerr=single_trip_latencies_client_std_subset,
                                    marker='o', label=f"{category_label} Client")
                axs[1, 1].errorbar(msg_sizes_subset, fabric_latencies_client_subset, yerr=fabric_latencies_client_std_subset,
                                    marker='o', label=f"{category_label} Client")
            if not ignore_server:
                axs[1, 0].errorbar(msg_sizes_subset, single_trip_latencies_server_subset, yerr=single_trip_latencies_server_std_subset,
                                    marker='o', label=f"{category_label} Server")
                axs[1, 1].errorbar(msg_sizes_subset, fabric_latencies_server_subset, yerr=fabric_latencies_server_std_subset,
                                    marker='o', label=f"{category_label} Server")

    for i in [0, 1]:
        for j in [0, 1]:
            axs[i, j].set_xscale('log', base=2)
            axs[i, j].set_yscale('log')
            axs[i, j].xaxis.set_major_formatter(plt.FuncFormatter(format_bytes))
            axs[i, j].yaxis.set_major_formatter(plt.FuncFormatter(time_format))
            axs[i, j].legend()

    axs[0, 1].set_yscale('log', base=2)
    axs[0, 1].yaxis.set_major_formatter(plt.FuncFormatter(format_bytes))

    axs[0, 0].set_title('Round-trip Latency vs Message Size')
    axs[0, 0].set_xlabel('Message Size (bytes)')
    axs[0, 0].set_ylabel('Round-trip Latency')

    axs[0, 1].set_title('Round-trip Throughput vs Message Size')
    axs[0, 1].set_xlabel('Message Size (bytes)')
    axs[0, 1].set_ylabel('Round-trip Throughput (bytes/s)')

    axs[1, 0].set_title('Single-trip Latency vs Message Size')
    axs[1, 0].set_xlabel('Message Size (bytes)')
    axs[1, 0].set_ylabel('Single-trip Latency')

    axs[1, 1].set_title('Fabric Latency vs Message Size')
    axs[1, 1].set_xlabel('Message Size (bytes)')
    axs[1, 1].set_ylabel('Fabric Latency')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'cross_block_comparison_{pairs}_metrics.png', dpi=500)
    plt.close()


if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="../data/", help="Directory containing device data folders")
    parser.add_argument("--ignore-client", action='store_true', help="Don't plot client single-trip and fabric latencies")
    parser.add_argument("--ignore-server", action='store_true', help="Don't plot server single-trip and fabric latencies")
    parser.add_argument("--pairs", type=int, nargs='+', default=[1],
                        help="List of num_pairs to include in the graphs (e.g. --pairs 1 2 4). If omitted, include only [1].")
    args = parser.parse_args()

    hwd_alias = {
        "NVIDIA_GeForce_RTX_5090": "5090",
        "NVIDIA_A100-SXM4-40GB": "A100",
        "NVIDIA_GeForce_RTX_3050_Ti_Laptop_GPU": "3050ti",
    }

    # Extracting device metrics
    devices = [d for d in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, d))]
    metrics = {}
    for device in devices:
        device_path = str(os.path.join(args.data_dir, device))
        json_path   = device_path + "_metrics.json"

        print(f"Loading metrics from {json_path}")
        with open(json_path, 'r') as f:
            device_metrics   = json.load(f)
            filtered_metrics = [m for m in device_metrics if m['num_pairs'] in args.pairs]
            metrics[hwd_alias.get(device, device)]  = filtered_metrics

    # Plotting comparison graphs
    print("Producing comparison plots...")
    plot_comparison_graphs(metrics, args.pairs, args.ignore_client, args.ignore_server)
