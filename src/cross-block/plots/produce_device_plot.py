import matplotlib.pyplot as plt
import numpy as np
import argparse
import json
import os


def compute_metrics_pair(pair_entries, msg_size):
    invalid_data = False

    client_recv_starts  = [entry["client_recv_start"] for entry in pair_entries.values()]
    client_recv_ends    = [entry["client_recv_end"] for entry in pair_entries.values()]
    client_trans_starts = [entry["client_trans_start"] for entry in pair_entries.values()]
    client_trans_ends   = [entry["client_trans_end"] for entry in pair_entries.values()]

    invalid_data |= any(ts == 0 for ts in client_recv_starts)
    invalid_data |= any(te == 0 for te in client_recv_ends)
    invalid_data |= any(ts == 0 for ts in client_trans_starts)
    invalid_data |= any(te == 0 for te in client_trans_ends)

    server_recv_starts  = [entry["server_recv_start"] for entry in pair_entries.values()]
    server_recv_ends    = [entry["server_recv_end"] for entry in pair_entries.values()]
    server_trans_starts = [entry["server_trans_start"] for entry in pair_entries.values()]
    server_trans_ends   = [entry["server_trans_end"] for entry in pair_entries.values()]

    invalid_data |= any(ts == 0 for ts in server_recv_starts)
    invalid_data |= any(te == 0 for te in server_recv_ends)
    invalid_data |= any(ts == 0 for ts in server_trans_starts)
    invalid_data |= any(te == 0 for te in server_trans_ends)

    if invalid_data:
        return (float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'))

    round_trip_latency = max(client_recv_ends) - min(client_trans_starts)
    if round_trip_latency <= 0:
        return (float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'))
    
    round_trip_throughput      = 2 * msg_size / (round_trip_latency / 1e9)  # bytes per second (ns -> s)
    single_trip_latency_client = max(server_recv_ends) - min(client_trans_starts)
    single_trip_latency_server = max(client_recv_ends) - min(server_trans_starts)
    send_overhead_client       = min(server_recv_starts) - min(client_trans_starts)
    send_overhead_server       = min(client_recv_starts) - min(server_trans_starts)
    fabric_latency_client      = max(min(server_recv_starts) - max(client_trans_ends), 0)
    fabric_latency_server      = max(min(client_recv_starts) - max(server_trans_ends), 0)

    return (round_trip_latency, round_trip_throughput, single_trip_latency_client, single_trip_latency_server,
            send_overhead_client, send_overhead_server, fabric_latency_client, fabric_latency_server)


def compute_metrics(json_path):
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

    invalid_runs = 0
    for run_idx in range(num_runs):
        pair_entries           = data[f"run{run_idx}"]
        m1, m2, m3, m4, m5, m6, m7, m8 = compute_metrics_pair(pair_entries, msg_size)

        if np.isnan(m1) or np.isnan(m2) or np.isnan(m3) or np.isnan(m4) or np.isnan(m5) or np.isnan(m6) or np.isnan(m7) or np.isnan(m8):
            invalid_runs += 1
            continue

        metrics_intermediate["round_trip_latency"].append(m1)
        metrics_intermediate["round_trip_throughput"].append(m2)
        metrics_intermediate["single_trip_latency_client"].append(m3)
        metrics_intermediate["single_trip_latency_server"].append(m4)
        metrics_intermediate["send_overhead_client"].append(m5)
        metrics_intermediate["send_overhead_server"].append(m6)
        metrics_intermediate["fabric_latency_client"].append(m7)
        metrics_intermediate["fabric_latency_server"].append(m8)

    if invalid_runs > 0:
        print(f"\033[91mWarning: Invalid data detected in {json_path}, {invalid_runs} runs skipped.\033[0m")

    metrics = {}
    for key in metrics_intermediate:
        values = metrics_intermediate[key]
        metrics[key + "_avg"] = np.mean(values)
        metrics[key + "_std"] = np.std(values)
    return msg_size, num_pairs, num_runs - invalid_runs, metrics


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
    fig.suptitle('Cross-Warp Benchmark Results on ' + device_name, fontsize=16)
    fig.delaxes(axs[0, 2])  # Remove unused subplot

    msg_sizes                    = np.array([entry["msg_size"] for entry in output_data])
    num_pairs                    = np.array([entry["num_pairs"] for entry in output_data])
    round_trip_latencies         = np.array([entry["metrics"]["round_trip_latency_avg"] for entry in output_data])
    round_trip_throughputs       = np.array([entry["metrics"]["round_trip_throughput_avg"] for entry in output_data])
    single_trip_latencies_client = np.array([entry["metrics"]["single_trip_latency_client_avg"] for entry in output_data])
    single_trip_latencies_server = np.array([entry["metrics"]["single_trip_latency_server_avg"] for entry in output_data])
    send_overheads_client        = np.array([entry["metrics"]["send_overhead_client_avg"] for entry in output_data])
    send_overheads_server        = np.array([entry["metrics"]["send_overhead_server_avg"] for entry in output_data])
    fabric_latencies_client      = np.array([entry["metrics"]["fabric_latency_client_avg"] for entry in output_data])
    fabric_latencies_server      = np.array([entry["metrics"]["fabric_latency_server_avg"] for entry in output_data])

    round_trip_latencies_std         = np.array([entry["metrics"]["round_trip_latency_std"] for entry in output_data])
    round_trip_throughputs_std       = np.array([entry["metrics"]["round_trip_throughput_std"] for entry in output_data])
    single_trip_latencies_client_std = np.array([entry["metrics"]["single_trip_latency_client_std"] for entry in output_data])
    single_trip_latencies_server_std = np.array([entry["metrics"]["single_trip_latency_server_std"] for entry in output_data])
    send_overheads_client_std        = np.array([entry["metrics"]["send_overhead_client_std"] for entry in output_data])
    send_overheads_server_std        = np.array([entry["metrics"]["send_overhead_server_std"] for entry in output_data])
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
        send_overheads_client_subset        = send_overheads_client[mask]
        send_overheads_server_subset        = send_overheads_server[mask]
        fabric_latencies_client_subset      = fabric_latencies_client[mask]
        fabric_latencies_server_subset      = fabric_latencies_server[mask]

        round_trip_latencies_std_subset         = round_trip_latencies_std[mask]
        round_trip_throughputs_std_subset       = round_trip_throughputs_std[mask]
        single_trip_latencies_client_std_subset = single_trip_latencies_client_std[mask]
        single_trip_latencies_server_std_subset = single_trip_latencies_server_std[mask]
        send_overheads_client_std_subset        = send_overheads_client_std[mask]
        send_overheads_server_std_subset        = send_overheads_server_std[mask]
        fabric_latencies_client_std_subset      = fabric_latencies_client_std[mask]
        fabric_latencies_server_std_subset      = fabric_latencies_server_std[mask]

        category_label = device_name + f" (P={pairs})"
        axs[0, 0].errorbar(msg_sizes_subset, round_trip_latencies_subset, yerr=round_trip_latencies_std_subset,
                            marker='o', label=f"{category_label}")
        axs[0, 1].errorbar(msg_sizes_subset, round_trip_throughputs_subset, yerr=round_trip_throughputs_std_subset,
                            marker='o', label=f"{category_label}")

        if not args.ignore_client:
            axs[1, 0].errorbar(msg_sizes_subset, single_trip_latencies_client_subset, yerr=single_trip_latencies_client_std_subset,
                                marker='o', label=f"{category_label} Client")
            axs[1, 1].errorbar(msg_sizes_subset, send_overheads_client_subset, yerr=send_overheads_client_std_subset,
                                marker='o', label=f"{category_label} Client")
            yerr = None if args.ignore_fabric_std else fabric_latencies_client_std_subset
            axs[1, 2].errorbar(msg_sizes_subset, fabric_latencies_client_subset, yerr=yerr, marker='o', label=f"{category_label} Client")

        if not args.ignore_server:
            axs[1, 0].errorbar(msg_sizes_subset, single_trip_latencies_server_subset, yerr=single_trip_latencies_server_std_subset,
                                marker='o', label=f"{category_label} Server")
            axs[1, 1].errorbar(msg_sizes_subset, send_overheads_server_subset, yerr=send_overheads_server_std_subset,
                                marker='o', label=f"{category_label} Server")
            yerr = None if args.ignore_fabric_std else fabric_latencies_server_std_subset
            axs[1, 2].errorbar(msg_sizes_subset, fabric_latencies_server_subset, yerr=yerr, marker='o', label=f"{category_label} Server")

    for i in [0, 1]:
        for j in [0, 1, 2]:
            if (i == 0 and j == 2): continue
            axs[i, j].set_xscale('log', base=2)
            axs[i, j].set_yscale('log') if not (i == 1 and j == 2) else axs[i, j].set_yscale('symlog')
            axs[i, j].xaxis.set_major_formatter(plt.FuncFormatter(format_bytes))
            axs[i, j].yaxis.set_major_formatter(plt.FuncFormatter(time_format))
            axs[i, j].legend()

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

    axs[1, 2].yaxis.set_minor_formatter(plt.FuncFormatter(time_format))
    axs[1, 2].set_title('Fabric Latency vs Message Size')
    axs[1, 2].set_xlabel('Message Size (bytes)')
    axs[1, 2].set_ylabel('Fabric Latency')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'cross_block_{device_name}_{selected_pairs}_metrics.png', dpi=500)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="../data/", help="Directory containing device data folders")
    parser.add_argument("--ignore-client", action='store_true', help="Don't plot client single-trip and fabric latencies")
    parser.add_argument("--ignore-server", action='store_true', help="Don't plot server single-trip and fabric latencies")
    parser.add_argument("--ignore-fabric-std", action='store_true', default=True, help="Don't plot fabric latency stddev")
    parser.add_argument("--pairs", type=int, nargs='+', default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
                        help="List of num_pairs to include in the graphs (e.g. --pairs 1 2 4). If omitted, include all.")
    args = parser.parse_args()

    devices   = [d for d in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, d))]
    hwd_alias = {
        "NVIDIA_GeForce_RTX_5090": "5090",
        "NVIDIA_A100-SXM4-40GB": "A100",
        "NVIDIA_GeForce_RTX_3050_Ti_Laptop_GPU": "3050ti",
    }

    for device in devices:
        device_path      = os.path.join(args.data_dir, device)
        json_files       = [f for f in os.listdir(device_path) if f.endswith('.json')]
        output_json_path = os.path.join(args.data_dir, f"{device}_metrics.json")
        output_data      = []

        for json_file in json_files:
            json_path = os.path.join(device_path, json_file)
            msg_size, num_pairs, num_runs, metrics = compute_metrics(json_path)
            output_data.append({
                'msg_size': msg_size, 'num_pairs': num_pairs,
                'num_runs': num_runs, 'metrics': metrics
            })

        output_data = sorted(output_data, key=lambda i: (i['msg_size'], i['num_pairs']))
        with open(output_json_path, 'w') as f:
            json.dump(output_data, f, indent=4)

        plot_device_metrics(hwd_alias.get(device, device), output_data, args.pairs, args)
