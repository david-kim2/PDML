import re
import matplotlib.pyplot as plt
import numpy as np
import argparse
import json
import os


def compute_metrics_pair(pair_entries):
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

    if invalid_data: return float('nan'), float('nan')
    return max(server_recv_ends) <= min(server_trans_starts), max(server_recv_ends) - min(server_trans_starts)


def compute_metrics(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    num_runs  = data["n_runs"]
    msg_size  = data["message_size"]
    num_pairs = data["num_pairs"]

    sync_count   = 0
    invalid_runs = 0
    max_diff     = 0
    for run_idx in range(num_runs):
        pair_entries = data[f"run{run_idx}"]
        sync, diff = compute_metrics_pair(pair_entries)
        max_diff = max(max_diff, diff)

        if np.isnan(sync):
            print(f"\033[91m\tWarning: Invalid data detected in run {run_idx}, skipping this run.\033[0m")
            invalid_runs += 1
            continue
        sync_count += int(sync)

    valid_runs   = num_runs - invalid_runs
    sync_percent = (sync_count / valid_runs) * 100 if valid_runs > 0 else 0.0
    return msg_size, num_pairs, valid_runs, sync_percent, max_diff


if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, required=True, choices=["5090", "A100", "3050ti"], help="Device to plot data for")
    parser.add_argument("--area", type=int, nargs='+', default=[1, 2, 3],
                        help="List of hardware areas to include in the metrics (e.g. --area 1 2 3). If omitted, include first three areas.")
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

    metrics = {area_alias.get(area) : [] for area in args.area}
    for area in args.area:
        device_name = reverse_hwd_alias.get(args.device, args.device)
        area_name   = area_alias.get(area, f"area_{area}")

        area_path   = os.path.join("..", area_name, "data")
        device_path = os.path.join(area_path, device_name)
        if not os.path.exists(device_path):
            print(f"Error: Device path {device_path} does not exist. Skipping area {area_name}.")
            continue
        json_files  = [f for f in os.listdir(device_path) if f.endswith('.json')]
        total_pairs = set()

        for json_file in json_files:
            json_path = os.path.join(device_path, json_file)
            msg_size, num_pairs, num_runs, sync_percent, max_diff = compute_metrics(json_path)
            total_pairs.add(num_pairs)

            metrics[area_name].append({
                "msg_size": msg_size, "num_pairs": num_pairs,
                "num_runs": num_runs, "sync_percent": sync_percent, "max_diff": max_diff
            })

        print(f"Area: {area_name}")
        for pair in sorted(list(total_pairs)):
            pair_metrics = [m["sync_percent"] for m in metrics[area_name] if m["num_pairs"] == pair]
            avg_sync = np.mean(pair_metrics) if pair_metrics else 0.0
            min_sync = np.min(pair_metrics) if pair_metrics else 0.0
            max_sync = np.max(pair_metrics) if pair_metrics else 0.0
            print(f"Num Pairs: {pair}, Average Sync: {avg_sync:.2f}%, Min Sync: {min_sync:.2f}%, Max Sync: {max_sync:.2f}%")
        print("\n")

    # Saving metrics to JSON
    def compact_metric_dicts(json_str):
        pattern = re.compile(r'\{\n\s+"msg_size":.*?\n\s+\}', re.DOTALL)
        def repl(match):
            inner = re.sub(r'\s+', ' ', match.group(0))
            inner = re.sub(r',\s*', ', ', inner)
            return inner
        return pattern.sub(repl, json_str)
        
    os.makedirs('data/sync_percent', exist_ok=True)
    output_path = f'data/sync_percent/{args.device}_areas_{args.area}.json'
    for area, area_metrics in metrics.items():
        area_metrics.sort(key=lambda x: (x["num_pairs"], x["msg_size"]))
        
    with open(output_path, 'w') as f:
        json_str      = json.dumps(metrics, indent=4)
        compacted_str = compact_metric_dicts(json_str)
        f.write(compacted_str)
    print(f"Saved sync percentage metrics to {output_path}")
