import numpy as np
import argparse
import json
import os


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

            client_fabric_latencies = [m['metrics']['fabric_latency_client_avg'] for m in device_metrics if m['num_pairs'] in args.pairs]
            client_single_latencies = [m['metrics']['single_trip_latency_server_avg'] for m in device_metrics if m['num_pairs'] in args.pairs]
            client_ratios           = [f / s if s != 0 else 0 for f, s in zip(client_fabric_latencies, client_single_latencies)]
            avg_client_ratio        = np.mean(client_ratios) if client_ratios else 0
            std_client_ratio        = np.std(client_ratios) if client_ratios else 0

            metrics.append({
                'area': area_name, 'device': args.device, 'type': 'client',
                'ratios': client_ratios, 'avg_ratio': avg_client_ratio, 'std_ratio': std_client_ratio
            })
            print(f"Avg Client Fabric to Single-Trip Latency Ratio for {args.device} in {area_name}: {avg_client_ratio:.4f}")
            print(f"Std Client Fabric to Single-Trip Latency Ratio for {args.device} in {area_name}: {std_client_ratio:.4f}")

            server_fabric_latencies = [m['metrics']['fabric_latency_server_avg'] for m in device_metrics if m['num_pairs'] in args.pairs]
            server_single_latencies = [m['metrics']['single_trip_latency_client_avg'] for m in device_metrics if m['num_pairs'] in args.pairs]
            server_ratios           = [f / s if s != 0 else 0 for f, s in zip(server_fabric_latencies, server_single_latencies)]
            avg_server_ratio        = np.mean(server_ratios) if server_ratios else 0
            std_server_ratio        = np.std(server_ratios) if server_ratios else 0

            metrics.append({
                'area': area_name, 'device': args.device, 'type': 'server',
                'ratios': server_ratios, 'avg_ratio': avg_server_ratio, 'std_ratio': std_server_ratio
            })
            print(f"Avg Server Fabric to Single-Trip Latency Ratio for {args.device} in {area_name}: {avg_server_ratio:.4f}")
            print(f"Std Server Fabric to Single-Trip Latency Ratio for {args.device} in {area_name}: {std_server_ratio:.4f}")
            print("")

    # Saving fabric ratios to JSON
    os.makedirs('data/fabric_ratios', exist_ok=True)
    output_path = f'data/fabric_ratios/{args.device}_areas_{args.area}_pairs_{args.pairs}.json'
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Fabric ratios saved to {output_path}")
