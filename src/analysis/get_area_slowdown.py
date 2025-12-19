from ast import pattern
import numpy as np
import argparse
import json
import os
import re


def format_bytes(x, pos):
    if x >= 1 << 30:   return f"{x / (1 << 30):.2f}GiB"
    elif x >= 1 << 20: return f"{x / (1 << 20):.2f}MiB"
    elif x >= 1 << 10: return f"{x / (1 << 10):.2f}KiB"
    else:              return f"{int(x)}B"


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

    # Extracting fabric ratios
    intermediate_metrics = []
    metrics              = []
    num_pairs            = None
    args.area.sort()

    for area in args.area:
        device_name = reverse_hwd_alias.get(args.device, args.device)
        area_name   = area_alias.get(area, f"area_{area}")

        area_path   = str(os.path.join("..", area_name, "data"))
        device_path = str(os.path.join(area_path, reverse_hwd_alias.get(args.device, args.device)))
        json_path   = device_path + "_metrics.json"

        print(f"Processing {json_path}...")
        with open(json_path, 'r') as f:
            device_metrics = json.load(f)
            intermediate_metrics.append((area_name, device_metrics))

            unique_num_pairs = set([m["num_pairs"] for m in device_metrics])
            num_pairs        = unique_num_pairs if num_pairs is None else num_pairs.union(unique_num_pairs)
    print(f"Union num_pairs across areas: {sorted(num_pairs)}")

    for pair in sorted(num_pairs):
        print(f"\nProcessing num_pairs = {pair}:")
        data_dict = {"pair": pair, "slowdowns": [], "slowdown_stats": []}

        for (area_name, device_metrics), (next_area_name, next_device_metrics) in zip(intermediate_metrics, intermediate_metrics[1:]):
            throughputs1 = [(m["msg_size"], m["metrics"]["round_trip_throughput_avg"]) for m in device_metrics if m["num_pairs"] == pair \
                                and ("mode" not in m or m["mode"] == "cda")]
            throughputs2 = [(m["msg_size"], m["metrics"]["round_trip_throughput_avg"]) for m in next_device_metrics if m["num_pairs"] == pair \
                                and ("mode" not in m or m["mode"] == "cda")]
            if len(throughputs1) == 0 or len(throughputs2) == 0:
                print(f"Skipping area pair {area_name}->{next_area_name} due to no data for num_pairs={pair}")
                continue

            peak_size1, peak_throughput1 = max(throughputs1, key=lambda x: x[1])
            peak_size2, peak_throughput2 = max(throughputs2, key=lambda x: x[1])

            msg_sizes1       = set(msg_size for msg_size, _ in throughputs1)
            msg_sizes2       = set(msg_size for msg_size, _ in throughputs2)
            common_msg_sizes = msg_sizes1.intersection(msg_sizes2)

            throughputs1_filtered = [(msg_size, throughput) for msg_size, throughput in throughputs1 if msg_size in common_msg_sizes]
            throughputs1_filtered = sorted(throughputs1_filtered, key=lambda x: x[0])
            throughputs2_filtered = [(msg_size, throughput) for msg_size, throughput in throughputs2 if msg_size in common_msg_sizes]
            throughputs2_filtered = sorted(throughputs2_filtered, key=lambda x: x[0])

            assert len(throughputs1_filtered) == len(throughputs2_filtered), \
                f"Mismatched throughput lengths {len(throughputs1_filtered)} vs {len(throughputs2_filtered)}"
            slowdowns      = [(msg_size, t2 / t1) for (msg_size, t1), (_, t2) in zip(throughputs1_filtered, throughputs2_filtered)]
            peak_slowdown  = peak_throughput2 / peak_throughput1
            just_slowdowns = [s for _, s in slowdowns]
            avg_slowdown   = np.mean(just_slowdowns)
            std_slowdown   = np.std(just_slowdowns)

            data_dict["slowdowns"].append({"areas": (area_name, next_area_name), "slowdowns": slowdowns})
            data_dict["slowdown_stats"].append({
                "areas": (area_name, next_area_name), "peak_slowdown": peak_slowdown,
                "avg_slowdown": avg_slowdown, "std_slowdown": std_slowdown}
            )
            print(f"Areas: {area_name}->{next_area_name}, Num Pairs: {pair}, Avg slowdown: {avg_slowdown:.2f}, Std: {std_slowdown:.2f}")
            print(f"\tPeak Throughput1: {format_bytes(peak_throughput1, None)}/s at Msg Size: {format_bytes(peak_size1, None)}")
            print(f"\tPeak Throughput2: {format_bytes(peak_throughput2, None)}/s at Msg Size: {format_bytes(peak_size2, None)}")
            print(f"\tPeak Slowdown: {peak_slowdown:.2f}")
        print("")
        metrics.append(data_dict)


    # Saving slowdown data to JSON
    def compact_inner_slowdown_arrays(json_str):
        pattern = re.compile(r'\[\s*([0-9.eE+-]+)\s*,\s*([0-9.eE+-]+)\s*\]', re.MULTILINE)
        return pattern.sub(r'[\1, \2]', json_str)

    def compact_areas_lists(json_str: str) -> str:
        def repl(match):
            inner = match.group(1)
            inner = re.sub(r'\s+', ' ', inner)
            inner = re.sub(r',\s*', ', ', inner)
            return f'"areas": [{inner.strip()}]'
        return re.sub(r'"areas"\s*:\s*\[\s*([\s\S]*?)\s*\]', repl, json_str)

    os.makedirs('data/area_slowdown', exist_ok=True)
    output_path = f'data/area_slowdown/{args.device}_areas_{args.area}.json'

    with open(output_path, 'w') as f:
        json_str = json.dumps(metrics, indent=4)
        compacted_json_str = compact_inner_slowdown_arrays(json_str)
        compacted_json_str = compact_areas_lists(compacted_json_str)
        f.write(compacted_json_str)
    print(f"Saved area slowdown data to {output_path}")
