import matplotlib.pyplot as plt
import numpy as np
import argparse
import json
import os


def compute_metrics_pair(pair_entries, msg_size):
    invalid_data = False

    client_recv_starts  = pair_entries["client_recv_start"]
    client_recv_ends    = pair_entries["client_recv_end"]
    client_trans_starts = pair_entries["client_trans_start"]
    client_trans_ends   = pair_entries["client_trans_end"]

    invalid_data |= client_recv_starts == 0
    invalid_data |= client_recv_ends == 0
    invalid_data |= client_trans_starts == 0
    invalid_data |= client_trans_ends == 0

    if invalid_data:
        return (float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'))

    round_trip_latency = client_recv_ends - client_trans_starts
    if round_trip_latency <= 0:
        return (float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'))
    
    round_trip_throughput = 2 * msg_size / (round_trip_latency / 1e9)  # bytes per second (ns -> s)

    return (round_trip_latency, round_trip_throughput)


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
    }

    invalid_runs = 0
    for run_idx in range(num_runs):
        pair_entries = data[f"run{run_idx}"]
        m1, m2       = compute_metrics_pair(pair_entries, msg_size)

        if np.isnan(m1) or np.isnan(m2):
            invalid_runs += 1
            continue

        metrics_intermediate["round_trip_latency"].append(m1)
        metrics_intermediate["round_trip_throughput"].append(m2)

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
    fig, axs = plt.subplots(1, 2, figsize=(21, 10))
    fig.suptitle('Cross-Host Benchmark Results on ' + device_name, fontsize=16)

    msg_sizes                  = np.array([entry["msg_size"] for entry in output_data])
    num_pairs                  = np.array([entry["num_pairs"] for entry in output_data])
    round_trip_latencies       = np.array([entry["metrics"]["round_trip_latency_avg"] for entry in output_data])
    round_trip_throughputs     = np.array([entry["metrics"]["round_trip_throughput_avg"] for entry in output_data])
    modes                      = np.array([e["mode"] for e in output_data])

    round_trip_latencies_std   = np.array([entry["metrics"]["round_trip_latency_std"] for entry in output_data])
    round_trip_throughputs_std = np.array([entry["metrics"]["round_trip_throughput_std"] for entry in output_data])

    colors = {
        'gdr': 'green',
        'cda': 'grey'
    }

    for pairs in selected_pairs:
        for mode, style in [("gdr", "o-"), ("cda", "s--")]:
            mask = (num_pairs == pairs) & (modes == mode)
            if not np.any(mask):
                continue

            axs[0].errorbar(
                msg_sizes[mask],
                round_trip_latencies[mask],
                yerr=round_trip_latencies_std[mask],
                fmt=style,
                color=colors[mode],
                label=f"P={pairs} ({mode})"
            )

            axs[1].errorbar(
                msg_sizes[mask],
                round_trip_throughputs[mask],
                yerr=round_trip_throughputs_std[mask],
                fmt=style,
                color=colors[mode],
                label=f"P={pairs} ({mode})"
            )

    selected_pairs_str = '_'.join(map(str, selected_pairs))
    selected_pairs = selected_pairs_str if len(selected_pairs) < 10 else 'all'

    for j in [0, 1]:
        axs[j].set_xscale('log', base=2)
        axs[j].xaxis.set_major_formatter(plt.FuncFormatter(format_bytes))
        axs[j].legend()

    axs[0].set_yscale('log', base=10)
    axs[0].yaxis.set_major_formatter(plt.FuncFormatter(time_format))
    axs[0].set_title('Round-trip Latency vs Message Size')
    axs[0].set_xlabel('Message Size (bytes)')
    axs[0].set_ylabel('Round-trip Latency (ns)')

    axs[1].set_yscale('log', base=2)
    axs[1].yaxis.set_major_formatter(plt.FuncFormatter(format_bytes))
    axs[1].set_title('Round-trip Throughput vs Message Size')
    axs[1].set_xlabel('Message Size (bytes)')
    axs[1].set_ylabel('Round-trip Throughput (bytes/s)')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'cross_host_{device_name}_{selected_pairs}_metrics.png', dpi=500)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="../data/", help="Directory containing device data folders")
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
            mode = "gdr" if "gdr" in json_file.lower() else "cda"
            output_data.append({
                'msg_size': msg_size, 'num_pairs': num_pairs,
                'num_runs': num_runs, 'metrics': metrics,
                'mode': mode
            })

        output_data = sorted(output_data, key=lambda i: (i['msg_size'], i['num_pairs']))
        with open(output_json_path, 'w') as f:
            json.dump(output_data, f, indent=4)

        plot_device_metrics(hwd_alias.get(device, device), output_data, args.pairs, args)