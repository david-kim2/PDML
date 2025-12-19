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

def min_latency(json_data):
    min_latency = float('inf')
    min_latency_entry = None
    for entry in json_data:
        latency = entry["metrics"]["round_trip_latency_avg"]
        if latency < min_latency:
            min_latency = latency
            min_latency_entry = entry
    return min_latency, min_latency_entry

def max_throughput(json_data):
    max_throughput = 0
    max_throughput_entry = None
    for entry in json_data:
        throughput = entry["metrics"]["round_trip_throughput_avg"]
        if throughput > max_throughput:
            max_throughput = throughput
            max_throughput_entry = entry
    return max_throughput, max_throughput_entry

if __name__ == "__main__":
    # Argument parsing
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

    # Table header
    header = (
        f"{'Area':<15}"
        f"{'Hardware':<12}"
        f"{'Min Latency':<18}"
        f"{'Latency @ (P, Size)':<22}"
        f"{'Max Throughput (B/s)':<24}"
        f"{'Throughput @ (P, Size)':<26}"
    )
    print(header)
    print("-" * len(header))

    for area_id in area_alias:
        dir_name = f"../{area_alias[area_id]}/data"
        for hwd in reverse_hwd_alias:
            json_name = f"{dir_name}/{reverse_hwd_alias[hwd]}_metrics.json"
            if not os.path.exists(json_name):
                continue

            with open(json_name, "r") as f:
                json_data = json.load(f)

            min_lat, min_entry = min_latency(json_data)
            max_thr, max_entry = max_throughput(json_data)

            # print(
            #     f"{area_alias[area_id]:<15}"
            #     f"{hwd:<12}"
            #     f"{min_lat:<18.2f}"
            #     f"{f'(P={min_entry['num_pairs']}, {min_entry['msg_size']}B)':<22}"
            #     f"{max_thr:<24.2f}"
            #     f"{f'(P={max_entry['num_pairs']}, {max_entry['msg_size']}B)':<26}"
            # )
            print(
                f"{area_alias[area_id]:<15}"
                f"{hwd:<12}"
                f"{time_format(min_lat, None):<18}"
                f"{f'(P={min_entry['num_pairs']}, {format_bytes(min_entry['msg_size'], None)})':<22}"
                f"{format_bytes(max_thr, None):<24}"
                f"{f'(P={max_entry['num_pairs']}, {format_bytes(max_entry['msg_size'], None)})':<26}"
            )

