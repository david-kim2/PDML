import matplotlib.pyplot as plt
import numpy as np
import json
import os


def compute_metrics_pair(pair_entries, msg_size, num_pairs):
    assert num_pairs == 1, "Currently only supports single pair data analysis." # TODO: Extend to multi-pair analysis
    entry = pair_entries["pair0"]  # Analyze the first pair only

    round_trip_latency         = entry["client_recv_end"] - entry["client_trans_start"]
    round_trip_throughput      = msg_size / (round_trip_latency / 1e9) # bytes per second (ns -> s)
    single_trip_latency_client = entry["server_recv_end"] - entry["client_trans_start"]
    single_trip_latency_server = entry["client_recv_end"] - entry["server_trans_start"]
    fabric_latency_client      = entry["client_recv_end"] - entry["client_recv_start"]
    fabric_latency_server      = entry["server_recv_end"] - entry["server_recv_start"]

    return (round_trip_latency, round_trip_throughput, single_trip_latency_client,
            single_trip_latency_server, fabric_latency_client, fabric_latency_server)

def format_bytes(x, pos):
    if x >= 1 << 30: return f"{int(x / (1 << 30))}GB"
    elif x >= 1 << 20: return f"{int(x / (1 << 20))}MB"
    elif x >= 1 << 10: return f"{int(x / (1 << 10))}KB"
    else: return f"{int(x)}B"

def time_format(x, pos):
            if x >= 1e9: return f"{int(x / 1e9)}s"
            elif x >= 1e6: return f"{int(x / 1e6)}ms"
            elif x >= 1e3: return f"{int(x / 1e3)}µs"
            else:return f"{int(x)}ns"

def compute_metrics(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    num_runs  = data["n_runs"]
    msg_size  = data["message_size"]
    num_pairs = data["num_pairs"]

    metrics_intermediate = {
        "round_trip_latency": [],
        "round_trip_throughput": [],
        "single_trip_latency_client": [],
        "single_trip_latency_server": [],
        "fabric_latency_client": [],
        "fabric_latency_server": [],
    }

    pair_idx = 0  # Currently only supports single pair data analysis
    for run_idx in range(num_runs):        
        pair_entries           = data[f"run{run_idx}"]
        m1, m2, m3, m4, m5, m6 = compute_metrics_pair(pair_entries, msg_size, num_pairs)

        metrics_intermediate["round_trip_latency"].append(m1)
        metrics_intermediate["round_trip_throughput"].append(m2)
        metrics_intermediate["single_trip_latency_client"].append(m3)
        metrics_intermediate["single_trip_latency_server"].append(m4)
        metrics_intermediate["fabric_latency_client"].append(m5)
        metrics_intermediate["fabric_latency_server"].append(m6)
        
    metrics = {}
    for key in metrics_intermediate:
        values = metrics_intermediate[key]
        metrics[key + "_avg"] = np.mean(values)
        metrics[key + "_std"] = np.std(values)
    return metrics


if __name__ == "__main__":
    data_dir    = "data/"
    devices     = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    all_metrics = {}
    os.makedirs('plots', exist_ok=True)
    
    hwd_alias = {
        "NVIDIA_GeForce_RTX_5090": "RTX 5090",
    }

    for device in devices:
        device_path = os.path.join(data_dir, device)
        json_files = [f for f in os.listdir(device_path) if f.endswith('.json')]
        all_metrics[device] = []

        for json_file in json_files:
            num_pairs = int(json_file.split('_')[2].replace('P.json', ''))
            if num_pairs != 1: continue # Currently only process single pair data
            
            json_path = os.path.join(device_path, json_file)
            metrics   = compute_metrics(json_path)
            all_metrics[device].append((json_file, metrics))

        output_json_path = os.path.join('data', f"{device}_metrics.json")
        output_data = []
        
        for json_file, metrics in all_metrics[device]:
            with open(os.path.join(device_path, json_file), 'r') as f:
                data = json.load(f)
            msg_size  = data["message_size"]
            num_pairs = data["num_pairs"]
            
            output_data.append({
                'msg_size': msg_size,
                'num_pairs': num_pairs,
                'metrics': metrics
            })

        output_data = sorted(output_data, key=lambda i: (i['msg_size'], i['num_pairs']))
        with open(output_json_path, 'w') as f:
            json.dump(output_data, f, indent=4)

    
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Cross-Warp Benchmark Results on ' + hwd_alias.get(device, device), fontsize=16)
        msg_sizes                    = [entry["msg_size"] for entry in output_data]
        num_pairs                    = [entry["num_pairs"] for entry in output_data]
        round_trip_latencies         = [entry["metrics"]["round_trip_latency_avg"] for entry in output_data]
        round_trip_throughputs       = [entry["metrics"]["round_trip_throughput_avg"] for entry in output_data]
        single_trip_latencies_client = [entry["metrics"]["single_trip_latency_client_avg"] for entry in output_data]
        single_trip_latencies_server = [entry["metrics"]["single_trip_latency_server_avg"] for entry in output_data]
        fabric_latencies_client      = [entry["metrics"]["fabric_latency_client_avg"] for entry in output_data]
        fabric_latencies_server      = [entry["metrics"]["fabric_latency_server_avg"] for entry in output_data]

        round_trip_latencies_std         = [entry["metrics"]["round_trip_latency_std"] for entry in output_data]
        round_trip_throughputs_std       = [entry["metrics"]["round_trip_throughput_std"] for entry in output_data]
        single_trip_latencies_client_std = [entry["metrics"]["single_trip_latency_client_std"] for entry in output_data]
        single_trip_latencies_server_std = [entry["metrics"]["single_trip_latency_server_std"] for entry in output_data]
        fabric_latencies_client_std      = [entry["metrics"]["fabric_latency_client_std"] for entry in output_data]
        fabric_latencies_server_std      = [entry["metrics"]["fabric_latency_server_std"] for entry in output_data]

        category_label = hwd_alias.get(device, device) + f" (Pairs: {1 << 0})"  # Currently only single pair data
        axs[0, 0].errorbar(msg_sizes, round_trip_latencies, yerr=round_trip_latencies_std, marker='o', label=f"{category_label}")
        axs[0, 1].errorbar(msg_sizes, round_trip_throughputs, yerr=round_trip_throughputs_std, marker='o', label=f"{category_label}")
        axs[1, 0].errorbar(msg_sizes, single_trip_latencies_client, yerr=single_trip_latencies_client_std, marker='o', label=f"{category_label} Client")
        axs[1, 0].errorbar(msg_sizes, single_trip_latencies_server, yerr=single_trip_latencies_server_std, marker='o', label=f"{category_label} Server")
        axs[1, 1].errorbar(msg_sizes, fabric_latencies_client, yerr=fabric_latencies_client_std, marker='o', label=f"{category_label} Client")
        axs[1, 1].errorbar(msg_sizes, fabric_latencies_server, yerr=fabric_latencies_server_std, marker='o', label=f"{category_label} Server")

        for i in [0, 1]:
            for j in [0, 1]:
                axs[i, j].set_xscale('log', base=2)
                axs[i, j].set_yscale('log')
                axs[i, j].xaxis.set_major_formatter(plt.FuncFormatter(format_bytes))
                axs[i, j].yaxis.set_major_formatter(plt.FuncFormatter(time_format))
                axs[i, j].legend()

        # make 0, 1 y axis log 2 and format bytes
        axs[0, 1].set_yscale('log', base=2)
        axs[0, 1].yaxis.set_major_formatter(plt.FuncFormatter(format_bytes))

        axs[0, 0].set_title('Round-trip Latency vs Message Size')
        axs[0, 0].set_xlabel('Message Size (bytes)')
        axs[0, 0].set_ylabel('Round-trip Latency')
        
        axs[0, 1].set_title('Round-trip Throughput vs Message Size')
        axs[0, 1].set_xlabel('Message Size (bytes)')
        axs[0, 1].set_ylabel('Round-trip Throughput (bytes/s)')
        
        axs[1, 0].set_title('Single-trip Latency (Client) vs Message Size')
        axs[1, 0].set_xlabel('Message Size (bytes)')
        axs[1, 0].set_ylabel('Single-trip Latency')
        
        axs[1, 1].set_title('Fabric Latency (Client) vs Message Size')
        axs[1, 1].set_xlabel('Message Size (bytes)')
        axs[1, 1].set_ylabel('Fabric Latency')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig('plots/cross_warp_benchmark_results.png', dpi=800)