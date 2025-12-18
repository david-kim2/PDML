import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import argparse
import json
import os


def compute_metrics_pair(pair_entries, msg_size, clock_freq_ghz=1.41):
    """
    Modified to convert GPU cycles to nanoseconds
    """
    # Convert cycles to nanoseconds
    def cycles_to_ns(cycles):
        return cycles / clock_freq_ghz

    client_recv_starts  = [cycles_to_ns(entry["client_recv_start"]) for entry in pair_entries.values()]
    client_recv_ends    = [cycles_to_ns(entry["client_recv_end"]) for entry in pair_entries.values()]
    client_trans_starts = [cycles_to_ns(entry["client_trans_start"]) for entry in pair_entries.values()]
    client_trans_ends   = [cycles_to_ns(entry["client_trans_end"]) for entry in pair_entries.values()]

    server_recv_starts  = [cycles_to_ns(entry["server_recv_start"]) for entry in pair_entries.values()]
    server_recv_ends    = [cycles_to_ns(entry["server_recv_end"]) for entry in pair_entries.values()]
    server_trans_starts = [cycles_to_ns(entry["server_trans_start"]) for entry in pair_entries.values()]
    server_trans_ends   = [cycles_to_ns(entry["server_trans_end"]) for entry in pair_entries.values()]

    # Calculate per-pair metrics
    fabric_latencies_client = [server_recv_starts[i] - client_trans_starts[i] for i in range(len(client_trans_starts))]
    fabric_latencies_server = [client_recv_starts[i] - server_trans_starts[i] for i in range(len(server_trans_starts))]

    round_trip_latency         = max(client_recv_ends) - min(client_trans_starts)
    round_trip_throughput      = 2 * msg_size / (round_trip_latency / 1e9)  # bytes per second (ns -> s)
    single_trip_latency_client = max(server_recv_ends) - min(client_trans_starts)
    single_trip_latency_server = max(client_recv_ends) - min(server_trans_starts)
    send_overhead_client       = min(client_trans_ends) - min(client_trans_starts)
    send_overhead_server       = min(server_trans_ends) - min(server_trans_starts)
    fabric_latency_client      = np.mean(fabric_latencies_client)
    fabric_latency_server      = np.mean(fabric_latencies_server)

    return (round_trip_latency, round_trip_throughput, single_trip_latency_client, single_trip_latency_server,
            send_overhead_client, send_overhead_server, fabric_latency_client, fabric_latency_server)


def compute_metrics(json_path, clock_freq_ghz=1.41):
    print(f"Processing {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)

    num_runs  = data["n_runs"]
    msg_size  = data["message_size"]
    num_pairs = data["num_pairs"]

    metrics_intermediate = {
        "round_trip_latency":         [],
        "round_trip_throughput":      [],
        "single_trip_latency_client": [],
        "single_trip_latency_server": [],
        "send_overhead_client":       [],
        "send_overhead_server":       [],
        "fabric_latency_client":      [],
        "fabric_latency_server":      [],
    }

    for run_idx in range(num_runs):
        pair_entries           = data[f"run{run_idx}"]
        m1, m2, m3, m4, m5, m6, m7, m8 = compute_metrics_pair(pair_entries, msg_size, clock_freq_ghz)

        metrics_intermediate["round_trip_latency"].append(m1)
        metrics_intermediate["round_trip_throughput"].append(m2)
        metrics_intermediate["single_trip_latency_client"].append(m3)
        metrics_intermediate["single_trip_latency_server"].append(m4)
        metrics_intermediate["send_overhead_client"].append(m5)
        metrics_intermediate["send_overhead_server"].append(m6)
        metrics_intermediate["fabric_latency_client"].append(m7)
        metrics_intermediate["fabric_latency_server"].append(m8)

    metrics = {}
    for key in metrics_intermediate:
        values = metrics_intermediate[key]
        metrics[key + "_avg"] = np.mean(values)
        metrics[key + "_std"] = np.std(values)
    return msg_size, num_pairs, num_runs, metrics


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


def plot_device_metrics(device_name, output_data, selected_pairs, args):
    fig, axs = plt.subplots(2, 3, figsize=(21, 10))
    fig.suptitle('Cross-Node Benchmark Results on ' + device_name, fontsize=16)
    fig.delaxes(axs[0, 2])  # Remove unused subplot

    msg_sizes                    = np.array([entry["msg_size"] for entry in output_data])
    num_pairs                    = np.array([entry["num_pairs"] for entry in output_data])
    round_trip_latencies         = np.array([entry["metrics"]["round_trip_latency_avg"] for entry in output_data])
    round_trip_throughputs       = np.array([entry["metrics"]["round_trip_throughput_avg"] for entry in output_data])
    single_trip_latencies_client = np.array([entry["metrics"]["single_trip_latency_client_avg"] for entry in output_data])
    single_trip_latencies_server = np.array([entry["metrics"]["single_trip_latency_server_avg"] for entry in output_data])
    send_overhead_client         = np.array([entry["metrics"]["send_overhead_client_avg"] for entry in output_data])
    send_overhead_server         = np.array([entry["metrics"]["send_overhead_server_avg"] for entry in output_data])
    fabric_latencies_client      = np.array([entry["metrics"]["fabric_latency_client_avg"] for entry in output_data])
    fabric_latencies_server      = np.array([entry["metrics"]["fabric_latency_server_avg"] for entry in output_data])

    round_trip_latencies_std         = np.array([entry["metrics"]["round_trip_latency_std"] for entry in output_data])
    round_trip_throughputs_std       = np.array([entry["metrics"]["round_trip_throughput_std"] for entry in output_data])
    single_trip_latencies_client_std = np.array([entry["metrics"]["single_trip_latency_client_std"] for entry in output_data])
    single_trip_latencies_server_std = np.array([entry["metrics"]["single_trip_latency_server_std"] for entry in output_data])
    send_overhead_client_std         = np.array([entry["metrics"]["send_overhead_client_std"] for entry in output_data])
    send_overhead_server_std         = np.array([entry["metrics"]["send_overhead_server_std"] for entry in output_data])
    fabric_latencies_client_std      = np.array([entry["metrics"]["fabric_latency_client_std"] for entry in output_data])
    fabric_latencies_server_std      = np.array([entry["metrics"]["fabric_latency_server_std"] for entry in output_data])

    for pairs in selected_pairs:
        mask = (num_pairs == pairs)
        if not np.any(mask): continue

        msg_sizes_subset                    = msg_sizes[mask]
        round_trip_latencies_subset         = round_trip_latencies[mask]
        round_trip_throughputs_subset       = round_trip_throughputs[mask]
        single_trip_latencies_client_subset = single_trip_latencies_client[mask]
        single_trip_latencies_server_subset = single_trip_latencies_server[mask]
        send_overheads_client_subset         = send_overhead_client[mask]
        send_overheads_server_subset         = send_overhead_server[mask]
        fabric_latencies_client_subset      = fabric_latencies_client[mask]
        fabric_latencies_server_subset      = fabric_latencies_server[mask]

        round_trip_latencies_std_subset         = round_trip_latencies_std[mask]
        round_trip_throughputs_std_subset       = round_trip_throughputs_std[mask]
        single_trip_latencies_client_std_subset = single_trip_latencies_client_std[mask]
        single_trip_latencies_server_std_subset = single_trip_latencies_server_std[mask]
        send_overheads_client_std_subset         = send_overhead_client_std[mask]
        send_overheads_server_std_subset         = send_overhead_server_std[mask]
        fabric_latencies_client_std_subset      = fabric_latencies_client_std[mask]
        fabric_latencies_server_std_subset      = fabric_latencies_server_std[mask]

        # Rainbow colors based on position in selected_pairs list
        color_idx = selected_pairs.index(pairs)
        color = cm.rainbow(color_idx / max(len(selected_pairs) - 1, 1))
        
        category_label = device_name + f" (P={pairs})"
        axs[0, 0].errorbar(msg_sizes_subset, round_trip_latencies_subset, yerr=round_trip_latencies_std_subset,
                            marker='o', label=f"{category_label}", color=color)
        axs[0, 1].errorbar(msg_sizes_subset, round_trip_throughputs_subset, yerr=round_trip_throughputs_std_subset,
                            marker='o', label=f"{category_label}", color=color)

        if not args.ignore_client:
            axs[1, 0].errorbar(msg_sizes_subset, single_trip_latencies_client_subset, yerr=single_trip_latencies_client_std_subset,
                                marker='o', label=f"{category_label} Client", color=color)
            axs[1, 1].errorbar(msg_sizes_subset, send_overheads_client_subset, yerr=send_overheads_client_std_subset,
                                marker='o', label=f"{category_label} Client", color=color)
            yerr = None if args.ignore_fabric_std else fabric_latencies_client_std_subset
            axs[1, 2].errorbar(msg_sizes_subset, fabric_latencies_client_subset, yerr=yerr, marker='o', label=f"{category_label} Client", color=color)

        if not args.ignore_server:
            axs[1, 0].errorbar(msg_sizes_subset, single_trip_latencies_server_subset, yerr=single_trip_latencies_server_std_subset,
                                marker='o', label=f"{category_label} Server", color=color, linestyle='--')
            axs[1, 1].errorbar(msg_sizes_subset, send_overheads_server_subset, yerr=send_overheads_server_std_subset,
                                marker='o', label=f"{category_label} Server", color=color, linestyle='--')
            yerr = None if args.ignore_fabric_std else fabric_latencies_server_std_subset
            axs[1, 2].errorbar(msg_sizes_subset, fabric_latencies_server_subset, yerr=yerr, marker='o', label=f"{category_label} Server", color=color, linestyle='--')

    for i in [0, 1]:
        for j in [0, 1, 2]:
            if (i == 0 and j == 2): continue
            axs[i, j].set_xscale('log', base=2)
            axs[i, j].set_yscale('log')
            axs[i, j].xaxis.set_major_formatter(plt.FuncFormatter(format_bytes))
            axs[i, j].yaxis.set_major_formatter(plt.FuncFormatter(time_format))

    # Create a single shared legend
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.75, 0.90), fontsize=10)
    
    # Add note about line styles
    fig.text(0.68, 0.65, 'Solid: Client\nDashed: Server', fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.3))

    axs[0, 0].set_title('Round-trip Latency vs Message Size')
    axs[0, 0].set_xlabel('Message Size (bytes)')
    axs[0, 0].set_ylabel('Round-trip Latency')

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

    axs[1, 2].yaxis.set_major_formatter(plt.FuncFormatter(time_format))
    axs[1, 2].set_title('Network Latency vs Message Size')
    axs[1, 2].set_xlabel('Message Size (bytes)')
    axs[1, 2].set_ylabel('Network Latency')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'plots/cross_node_{device_name}_{selected_pairs}_metrics.png')
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data/", help="Directory containing device data folders")
    parser.add_argument("--ignore-client", action='store_true', help="Don't plot client single-trip and network latencies")
    parser.add_argument("--ignore-server", action='store_true', help="Don't plot server single-trip and network latencies")
    parser.add_argument("--ignore-fabric-std", action='store_true', default=True, help="Don't plot network latency stddev")
    parser.add_argument("--clock-freq", type=float, default=1.41, help="GPU clock frequency in GHz (default: 1.41 for A100)")
    parser.add_argument("--pairs", type=int, nargs='+', default=None,
                        help="List of num_pairs to include in the graphs (e.g. --pairs 1 2 4). Default: plot all available pairs")
    args = parser.parse_args()

    os.makedirs('plots', exist_ok=True)

    devices   = [d for d in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, d))]
    hwd_alias = {
        "NVIDIA_A100-SXM4-40GB": "A100",
        "NVIDIA_H100_80GB_HBM3": "H100",
    }

    for device in devices:
        device_path      = os.path.join(args.data_dir, device)
        json_files       = [f for f in os.listdir(device_path) if f.startswith('crossnode_metrics') and f.endswith('.json')]
        
        if not json_files:
            print(f"Warning: No crossnode_metrics JSON files found in {device_path}")
            continue
            
        output_json_path = os.path.join(args.data_dir, f"{device}_crossnode_metrics.json")
        output_data      = []

        for json_file in json_files:
            json_path = os.path.join(device_path, json_file)
            msg_size, num_pairs, num_runs, metrics = compute_metrics(json_path, args.clock_freq)
            output_data.append({
                'msg_size': msg_size, 'num_pairs': num_pairs,
                'num_runs': num_runs, 'metrics': metrics
            })

        output_data = sorted(output_data, key=lambda i: (i['msg_size'], i['num_pairs']))
        with open(output_json_path, 'w') as f:
            json.dump(output_data, f, indent=4)

        # If no specific pairs requested, use all available pairs
        pairs_to_plot = args.pairs if args.pairs is not None else sorted(set(entry['num_pairs'] for entry in output_data))
        
        plot_device_metrics(hwd_alias.get(device, device), output_data, pairs_to_plot, args)
        
        print(f"✓ Generated plot for {device} (pairs: {pairs_to_plot})")
        print(f"  Output: plots/cross_node_{hwd_alias.get(device, device)}_{pairs_to_plot}_metrics.png")