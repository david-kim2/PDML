import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import argparse
import json
import os


def format_bytes(x, pos):
    if x >= 1 << 30:   return f"{int(x / (1 << 30))}GiB"
    elif x >= 1 << 20: return f"{int(x / (1 << 20))}MiB"
    elif x >= 1 << 10: return f"{int(x / (1 << 10))}KiB"
    else:              return f"{int(x)}B"


def time_format(x, pos):
    if x >= 1e9:   return f"{int(x / 1e9)}s"
    elif x >= 1e6: return f"{int(x / 1e6)}ms"
    elif x >= 1e3: return f"{int(x / 1e3)}µs"
    else:          return f"{int(x)}ns"


def plot_area_graphs(metrics, device, pairs, args):
    fig, axs = plt.subplots(2, 3, figsize=(21, 10))
    fig.suptitle('Cross-Area Benchmark Results on ' + device, fontsize=16)
    fig.delaxes(axs[0, 2])  # Remove unused subplot

    for i, (area, area_metrics) in enumerate(metrics.items()):
        msg_sizes                    = np.array([entry["msg_size"] for entry in area_metrics \
                                            if "mode" not in entry or entry["mode"] == "cda"])
        num_pairs                    = np.array([entry["num_pairs"] for entry in area_metrics \
                                            if "mode" not in entry or entry["mode"] == "cda"])
        round_trip_latencies         = np.array([entry["metrics"]["round_trip_latency_avg"] for entry in area_metrics \
                                            if "mode" not in entry or entry["mode"] == "cda"])
        round_trip_throughputs       = np.array([entry["metrics"]["round_trip_throughput_avg"] for entry in area_metrics \
                                            if "mode" not in entry or entry["mode"] == "cda"])
        single_trip_latencies_client = np.array([entry["metrics"]["single_trip_latency_client_avg"] for entry in area_metrics]) \
                                            if area != "cross-host" else []
        single_trip_latencies_server = np.array([entry["metrics"]["single_trip_latency_server_avg"] for entry in area_metrics]) \
                                            if area != "cross-host" else []
        send_overheads_client        = np.array([entry["metrics"]["send_overhead_client_avg"] for entry in area_metrics]) \
                                            if area != "cross-host" else []
        send_overheads_server        = np.array([entry["metrics"]["send_overhead_server_avg"] for entry in area_metrics]) \
                                            if area != "cross-host" else []

        fabric_latencies_client = np.array([entry["metrics"]["fabric_latency_client_avg"] for entry in area_metrics \
                                    if entry["msg_size"] == entry["num_pairs"] and entry["num_pairs"] in pairs]) \
                                    if area != "cross-host" else []
        fabric_latencies_server = np.array([entry["metrics"]["fabric_latency_server_avg"] for entry in area_metrics \
                                    if entry["msg_size"] == entry["num_pairs"] and entry["num_pairs"] in pairs]) \
                                    if area != "cross-host" else []

        round_trip_latencies_std         = np.array([entry["metrics"]["round_trip_latency_std"] for entry in area_metrics \
                                            if "mode" not in entry or entry["mode"] == "cda"])
        round_trip_throughputs_std       = np.array([entry["metrics"]["round_trip_throughput_std"] for entry in area_metrics \
                                            if "mode" not in entry or entry["mode"] == "cda"])
        single_trip_latencies_client_std = np.array([entry["metrics"]["single_trip_latency_client_std"] for entry in area_metrics]) \
                                                if area != "cross-host" else []
        single_trip_latencies_server_std = np.array([entry["metrics"]["single_trip_latency_server_std"] for entry in area_metrics]) \
                                                if area != "cross-host" else []
        send_overheads_client_std        = np.array([entry["metrics"]["send_overhead_client_std"] for entry in area_metrics]) \
                                                if area != "cross-host" else []
        send_overheads_server_std        = np.array([entry["metrics"]["send_overhead_server_std"] for entry in area_metrics]) \
                                                if area != "cross-host" else []

        fabric_latencies_client_std      = np.array([entry["metrics"]["fabric_latency_client_std"] for entry in area_metrics \
                                    if entry["msg_size"] == entry["num_pairs"] and entry["num_pairs"] in pairs]) \
                                    if area != "cross-host" else []
        fabric_latencies_server_std      = np.array([entry["metrics"]["fabric_latency_server_std"] for entry in area_metrics \
                                    if entry["msg_size"] == entry["num_pairs"] and entry["num_pairs"] in pairs]) \
                                    if area != "cross-host" else []

        for pair in pairs:
            mask = (num_pairs == pair)
            if not np.any(mask): continue

            msg_sizes_subset                    = msg_sizes[mask]
            round_trip_latencies_subset         = round_trip_latencies[mask]
            round_trip_throughputs_subset       = round_trip_throughputs[mask]
            single_trip_latencies_client_subset = single_trip_latencies_client[mask] if area != "cross-host" else []
            single_trip_latencies_server_subset = single_trip_latencies_server[mask] if area != "cross-host" else []
            send_overheads_client_subset        = send_overheads_client[mask] if area != "cross-host" else []
            send_overheads_server_subset        = send_overheads_server[mask] if area != "cross-host" else []

            round_trip_latencies_std_subset         = round_trip_latencies_std[mask]
            round_trip_throughputs_std_subset       = round_trip_throughputs_std[mask]
            single_trip_latencies_client_std_subset = single_trip_latencies_client_std[mask] if area != "cross-host" else []
            single_trip_latencies_server_std_subset = single_trip_latencies_server_std[mask] if area != "cross-host" else []
            send_overheads_client_std_subset        = send_overheads_client_std[mask] if area != "cross-host" else []
            send_overheads_server_std_subset        = send_overheads_server_std[mask] if area != "cross-host" else []

            # Rainbow colors based on position in selected_pairs list
            color = cm.viridis(i / max(len(metrics) - 1, 1))

            category_label = area.title() + f" (P={pair})"
            axs[0, 0].errorbar(msg_sizes_subset, round_trip_latencies_subset, yerr=round_trip_latencies_std_subset,
                                marker='o', label=f"{category_label}", color=color)
            axs[0, 1].errorbar(msg_sizes_subset, round_trip_throughputs_subset, yerr=round_trip_throughputs_std_subset,
                                marker='o', label=f"{category_label}", color=color)

            if not args.ignore_client and area != "cross-host":
                axs[1, 0].errorbar(msg_sizes_subset, single_trip_latencies_client_subset, yerr=single_trip_latencies_client_std_subset,
                                    marker='o', label=f"{category_label} Client", color=color)
                axs[1, 1].errorbar(msg_sizes_subset, send_overheads_client_subset, yerr=send_overheads_client_std_subset,
                                    marker='o', label=f"{category_label} Client", color=color)

            if not args.ignore_server and area != "cross-host":
                axs[1, 0].errorbar(msg_sizes_subset, single_trip_latencies_server_subset, yerr=single_trip_latencies_server_std_subset,
                                    marker='o', label=f"{category_label} Server", color=color, linestyle='--')
                axs[1, 1].errorbar(msg_sizes_subset, send_overheads_server_subset, yerr=send_overheads_server_std_subset,
                                    marker='o', label=f"{category_label} Server", color=color, linestyle='--')

        if area != "cross-host":
            axs[1, 2].errorbar(pairs, fabric_latencies_client, yerr=fabric_latencies_client_std,
                                marker='o', label=f"{category_label} Client")
            axs[1, 2].errorbar(pairs, fabric_latencies_server, yerr=fabric_latencies_server_std,
                                marker='o', label=f"{category_label} Server", linestyle='--')

    for i in [0, 1]:
        for j in [0, 1, 2]:
            if (i == 0 and j == 2): continue
            axs[i, j].set_xscale('log', base=2)
            axs[i, j].set_yscale('log') if i != 1 or j != 2 else axs[i, j].set_yscale('linear')
            axs[i, j].xaxis.set_major_formatter(plt.FuncFormatter(format_bytes))
            axs[i, j].yaxis.set_major_formatter(plt.FuncFormatter(time_format))

    # Create a single shared legend
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.8, 0.875), fontsize=12)
    fig.text(0.69, 0.675, 'Solid: Client\nDashed: Server', fontsize=12, bbox=dict(boxstyle='round', facecolor='white', alpha=0.3))

    axs[0, 0].set_title('Round-trip Latency vs Message Size')
    axs[0, 0].set_xlabel('Message Size (bytes)')
    axs[0, 0].set_ylabel('Round-trip Latency')

    axs[0, 1].set_yscale('log', base=2)
    axs[0, 1].yaxis.set_major_formatter(plt.FuncFormatter(format_bytes))
    axs[0, 1].set_title('Round-trip Throughput vs Message Size')
    axs[0, 1].set_xlabel('Message Size (bytes)')
    axs[0, 1].set_ylabel('Round-trip Throughput (bytes/s)')

    axs[1, 0].set_title('Single-trip Latency vs Message Size')
    axs[1, 0].set_xlabel('Message Size (bytes)')
    axs[1, 0].set_ylabel('Single-trip Latency')

    axs[1, 1].set_title('Send Overhead vs Message Size')
    axs[1, 1].set_xlabel('Message Size (bytes)')
    axs[1, 1].set_ylabel('Send Overhead')

    axs[1, 2].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: int(x)))
    axs[1, 2].set_title('Fabric Latency vs # of Pairs')
    axs[1, 2].set_xlabel('# of Pairs')
    axs[1, 2].set_ylabel('Fabric Latency')

    reverse_area_alias = {
        "cross-thread": 1,
        "cross-warp": 2,
        "cross-block": 3,
        "cross-gpu": 4,
        "cross-node": 5,
        "cross-host": 6,
    }

    area_tags = [reverse_area_alias[area] for area in metrics.keys()]
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'plots/{device}_area_{area_tags}_pair_{pairs}_metrics.png', dpi=500)
    plt.close()


if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, required=True, choices=["5090", "A100", "3050ti"], help="Device to plot data for")
    parser.add_argument("--ignore-client", action='store_true', help="Don't plot client single-trip and fabric latencies")
    parser.add_argument("--ignore-server", action='store_true', help="Don't plot server single-trip and fabric latencies")
    parser.add_argument("--area", type=int, nargs='+', default=[1, 2, 3],
                        help="List of hardware areas to include in the plots (e.g. --area 1 2 3). If omitted, include first three areas.")
    parser.add_argument("--pairs", type=int, nargs='+', default=[1],
                        help="List of num_pairs to include in the graphs (e.g. --pairs 1 2 4). If omitted, include only [1].")
    args = parser.parse_args()

    reverse_hwd_alias = {
        "5090": "NVIDIA_GeForce_RTX_5090",
        "A100": "NVIDIA_A100-SXM4-40GB",
        "3050ti": "NVIDIA_GeForce_RTX_3050_Ti_Laptop_GPU",
    }

    area_alias = {
        1: "cross-thread",
        2: "cross-warp",
        3: "cross-block",
        4: "cross-gpu",
        5: "cross-node",
        6: "cross-host",
    }

    # Extracting device metrics
    metrics = {}
    for area in args.area:
        alias = area_alias.get(area, f"area_{area}")
        area_path   = str(os.path.join("..", alias, "data"))
        device_path = str(os.path.join(area_path, reverse_hwd_alias.get(args.device, args.device)))
        json_path   = device_path + "_metrics.json"

        print(f"Loading metrics from {json_path}")
        with open(json_path, 'r') as f:
            area_metrics     = json.load(f)
            filtered_metrics = [m for m in area_metrics if m['num_pairs'] in args.pairs]
            metrics[alias]   = filtered_metrics

    # Plotting area graphs
    print("Producing area plots...")
    os.makedirs('plots', exist_ok=True)
    plot_area_graphs(metrics, args.device, args.pairs, args)
