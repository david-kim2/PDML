import numpy as np
import itertools
import argparse
import json
import os
import re


if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, required=True, choices=["5090", "A100", "3050ti"], help="Device to plot data for")
    parser.add_argument("--area", type=int, nargs='+', default=[1, 2, 3],
                        help="List of hardware areas to include in the metrics (e.g. --area 1 2 3). If omitted, include first three areas. " \
                        "Never include area 6 as it is not relevant for speedup analysis.")
    args = parser.parse_args()
    assert 6 not in args.area, "Area 6 should not be included for speedup analysis."

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

    # Extracting pair speedup metrics
    metrics = []

    for area in args.area:
        device_name = reverse_hwd_alias.get(args.device, args.device)
        area_name   = area_alias.get(area, f"area_{area}")

        area_path   = str(os.path.join("..", area_name, "data"))
        device_path = str(os.path.join(area_path, reverse_hwd_alias.get(args.device, args.device)))
        json_path   = device_path + "_metrics.json"

        print(f"Processing {json_path}...")
        with open(json_path, 'r') as f:
            device_metrics = json.load(f)

            unique_num_pairs = set([m["num_pairs"] for m in device_metrics])
            unique_num_pairs = sorted(list(unique_num_pairs))
            data_dict = {"area": area_name, "device": args.device, "speedups": [], "speedup_stats": []}

            for pair1, pair2 in itertools.combinations(unique_num_pairs, 2):
                throughputs1 = [(m["msg_size"], m["metrics"]["round_trip_throughput_avg"]) for m in device_metrics if m["num_pairs"] == pair1]
                throughputs2 = [(m["msg_size"], m["metrics"]["round_trip_throughput_avg"]) for m in device_metrics if m["num_pairs"] == pair2]

                msg_sizes1       = set(msg_size for msg_size, _ in throughputs1)
                msg_sizes2       = set(msg_size for msg_size, _ in throughputs2)
                common_msg_sizes = msg_sizes1.intersection(msg_sizes2)

                throughputs1_filtered = [(msg_size, throughput) for msg_size, throughput in throughputs1 if msg_size in common_msg_sizes]
                throughputs1_filtered = sorted(throughputs1_filtered, key=lambda x: x[0])
                throughputs2_filtered = [(msg_size, throughput) for msg_size, throughput in throughputs2 if msg_size in common_msg_sizes]
                throughputs2_filtered = sorted(throughputs2_filtered, key=lambda x: x[0])

                assert len(throughputs1_filtered) == len(throughputs2_filtered), "Mismatched throughput lengths"
                speedups = [(msg_size, t2 / t1) for (msg_size, t1), (_, t2) in zip(throughputs1_filtered, throughputs2_filtered)]
                just_speedups = [s for _, s in speedups]
                avg_speedup = np.mean(just_speedups)
                std_speedup = np.std(just_speedups)

                data_dict["speedups"].append({"pairs": (pair1, pair2), "speedups": speedups})
                data_dict["speedup_stats"].append({"pairs": (pair1, pair2), "avg_speedup": avg_speedup, "std_speedup": std_speedup})
                print(f"Area: {area_name}, Device: {args.device}, Pairs: {pair1}->{pair2}, Avg Speedup: {avg_speedup:.2f}, Std: {std_speedup:.2f}")

            print("")
            metrics.append(data_dict)

    # Saving speedup data to JSON
    def compact_inner_speedup_arrays(json_str):
        pattern = re.compile(r'\[\s*([0-9.eE+-]+)\s*,\s*([0-9.eE+-]+)\s*\]', re.MULTILINE)
        return pattern.sub(r'[\1, \2]', json_str)

    def compact_pairs_lists(json_str: str) -> str:
        def repl(match):
            inner = match.group(1)
            inner = re.sub(r'\s+', ' ', inner)
            inner = re.sub(r',\s*', ', ', inner)
            return f'"pairs": [{inner.strip()}]'
        return re.sub(r'"pairs"\s*:\s*\[\s*([\s\S]*?)\s*\]', repl, json_str)

    os.makedirs('data/pair_speedup', exist_ok=True)
    output_path = f'data/pair_speedup/{args.device}_areas_{args.area}.json'

    with open(output_path, 'w') as f:
        json_str = json.dumps(metrics, indent=4)
        compacted_json_str = compact_inner_speedup_arrays(json_str)
        compacted_json_str = compact_pairs_lists(compacted_json_str)
        f.write(compacted_json_str)
    print(f"Saved pair speedup data to {output_path}")
