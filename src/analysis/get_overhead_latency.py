import numpy as np
import argparse
import json
import os
import re


if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, required=True, choices=["5090", "A100", "3050ti"], help="Device to plot data for")
    parser.add_argument("--area", type=int, nargs='+', default=[1, 2, 3],
                        help="List of hardware areas to include in the metrics (e.g. --area 1 2 3). If omitted, include first three areas.")
    parser.add_argument("--pairs", type=int, nargs='+', default=[1],
                        help="List of num_pairs to include in the metrics (e.g. --pairs 1 2 4). If omitted, include only [1].")
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
    metrics = []

    for area in args.area:
        device_name = reverse_hwd_alias.get(args.device, args.device)
        area_name   = area_alias.get(area, f"area_{area}")

        area_path   = str(os.path.join("..", area_name, "data"))
        device_path = str(os.path.join(area_path, reverse_hwd_alias.get(args.device, args.device)))
        json_path   = device_path + "_metrics.json"

        with open(json_path, 'r') as f:
            device_metrics = json.load(f)

            client_num_pairs        = [m['num_pairs'] for m in device_metrics if m['num_pairs'] in args.pairs]
            client_send_overhead    = [m['metrics']['send_overhead_client_avg'] for m in device_metrics if m['num_pairs'] in args.pairs]
            client_single_latencies = [m['metrics']['single_trip_latency_server_avg'] for m in device_metrics if m['num_pairs'] in args.pairs]
            client_ratios           = [f / s if s != 0 else 0 for f, s in zip(client_send_overhead, client_single_latencies)]

            avg_client_ratio    = np.mean(client_ratios) if client_ratios else 0
            std_client_ratio    = np.std(client_ratios) if client_ratios else 0
            avg_client_overhead = np.mean(client_send_overhead) if client_send_overhead else 0
            std_client_overhead = np.std(client_send_overhead) if client_send_overhead else 0

            metrics.append({
                'area': area_name, 'device': args.device, 'type': 'client',
                'latencies': list(zip(client_num_pairs, client_send_overhead)),
                'avg_latency': avg_client_overhead, 'std_latency': std_client_overhead,
                'ratios': list(zip(client_num_pairs, client_ratios)),
                'avg_ratio': avg_client_ratio, 'std_ratio': std_client_ratio,
            })

            print(f"Avg Client Send Overhead to Single-Trip Latency Ratio for {args.device} in {area_name}: {avg_client_ratio:.4f}")
            print(f"Std Client Send Overhead to Single-Trip Latency Ratio for {args.device} in {area_name}: {std_client_ratio:.4f}")
            print(f"Avg Client Send Overhead for {args.device} in {area_name}: {avg_client_overhead:.4f} ns")
            print(f"Std Client Send Overhead for {args.device} in {area_name}: {std_client_overhead:.4f} ns\n")

            server_num_pairs        = [m['num_pairs'] for m in device_metrics if m['num_pairs'] in args.pairs]
            server_send_overhead    = [m['metrics']['send_overhead_server_avg'] for m in device_metrics if m['num_pairs'] in args.pairs]
            server_single_latencies = [m['metrics']['single_trip_latency_client_avg'] for m in device_metrics if m['num_pairs'] in args.pairs]
            server_ratios           = [f / s if s != 0 else 0 for f, s in zip(server_send_overhead, server_single_latencies)]

            avg_server_ratio    = np.mean(server_ratios) if server_ratios else 0
            std_server_ratio    = np.std(server_ratios) if server_ratios else 0
            avg_server_overhead = np.mean(server_send_overhead) if server_send_overhead else 0
            std_server_overhead = np.std(server_send_overhead) if server_send_overhead else 0

            metrics.append({
                'area': area_name, 'device': args.device, 'type': 'server',
                'latencies': list(zip(server_num_pairs, server_send_overhead)),
                'avg_latency': avg_server_overhead, 'std_latency': std_server_overhead,
                'ratios': list(zip(server_num_pairs, server_ratios)),
                'avg_ratio': avg_server_ratio, 'std_ratio': std_server_ratio
            })

            print(f"Avg Server Send Overhead to Single-Trip Latency Ratio for {args.device} in {area_name}: {avg_server_ratio:.4f}")
            print(f"Std Server Send Overhead to Single-Trip Latency Ratio for {args.device} in {area_name}: {std_server_ratio:.4f}")
            print(f"Avg Server Send Overhead for {args.device} in {area_name}: {avg_server_overhead:.4f} ns")
            print(f"Std Server Send Overhead for {args.device} in {area_name}: {std_server_overhead:.4f} ns\n")

    # Saving overhead latency data to JSON
    def compact_inner_arrays(json_str):
        pattern = re.compile(r'\[\s*([0-9.eE+-]+)\s*,\s*([0-9.eE+-]+)\s*\]', re.MULTILINE)
        return pattern.sub(r'[\1, \2]', json_str)

    os.makedirs('data/overhead_latency', exist_ok=True)
    output_path = f'data/overhead_latency/{args.device}_areas_{args.area}_pairs_{args.pairs}.json'

    with open(output_path, 'w') as f:
        json_str = json.dumps(metrics, indent=4)
        compacted_json_str = compact_inner_arrays(json_str)
        f.write(compacted_json_str)
    print(f"Saved overhead latency data to {output_path}")
