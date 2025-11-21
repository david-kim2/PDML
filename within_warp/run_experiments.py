import torch
from cuda.binding import warp_echo_forward
import json
import subprocess
import numpy as np
from tqdm import tqdm

def get_gpu_clock_freq_ghz():
    # Query SM clock in MHz
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=clocks.max.sm', '--format=csv,noheader,nounits'],
        capture_output=True,
        text=True,
        check=True
    )
    clock_mhz = float(result.stdout.strip().split("\n")[0])
    clock_ghz = clock_mhz / 1000.0
    return clock_ghz

def handle_timer_overflow(end_time, start_time):
    """
    Handle uint32 timer overflow.
    
    If end_time < start_time, the timer wrapped around.
    """
    if end_time >= start_time:
        return end_time - start_time
    else:
        max_uint32 = 2**32 - 1
        return (max_uint32 - start_time) + end_time + 1

def run_experiment(message_size_bytes, num_trials=100):
    """Run warp echo experiment for given message size"""
    device = torch.device('cuda')
    
    # Prepare message
    num_words = (message_size_bytes + 3) // 4
    
    message = torch.randint(0, 2**31 - 1, (num_words,), dtype=torch.uint32, device=device)
    
    latencies = []

    clock_freq_ghz = get_gpu_clock_freq_ghz()
    
    for _ in range(num_trials):
        results = warp_echo_forward(message, message_size_bytes)
        results_cpu = results.cpu().numpy()
        
        cl_tx_st, cl_tx_ed, cl_rx = results_cpu[0], results_cpu[1], results_cpu[2]
        sv_rx, sv_tx_st, sv_tx_ed = results_cpu[3], results_cpu[4], results_cpu[5]
        
        round_trip_cycles = handle_timer_overflow(cl_rx, cl_tx_st)
        single_trip_c2s_cycles = handle_timer_overflow(sv_rx, cl_tx_st)
        single_trip_s2c_cycles = handle_timer_overflow(cl_rx, sv_tx_st)
        send_overhead_client_cycles = handle_timer_overflow(cl_tx_ed, cl_tx_st)
        send_overhead_server_cycles = handle_timer_overflow(sv_tx_ed, sv_tx_st)
        
        # Convert cycles to nanoseconds
        round_trip_latency = round_trip_cycles / clock_freq_ghz
        single_trip_c2s = single_trip_c2s_cycles / clock_freq_ghz
        single_trip_s2c = single_trip_s2c_cycles / clock_freq_ghz
        send_overhead_client = send_overhead_client_cycles / clock_freq_ghz
        send_overhead_server = send_overhead_server_cycles / clock_freq_ghz
        
        latencies.append({
            'round_trip_ns': round_trip_latency,
            'single_trip_c2s_ns': single_trip_c2s,
            'single_trip_s2c_ns': single_trip_s2c,
            'send_overhead_client_ns': send_overhead_client,
            'send_overhead_server_ns': send_overhead_server
        })
    
    # Calculate statistics
    metrics = {}
    for key in latencies[0].keys():
        values = [l[key] for l in latencies]
        metrics[key] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values))
        }
    
    # Calculate throughput (GB/s)
    mean_latency_s = metrics['round_trip_ns']['mean'] / 1e9
    throughput_gbps = (message_size_bytes / mean_latency_s) / 1e9
    metrics['throughput_gbps'] = float(throughput_gbps)
    
    return metrics

def main():
    message_sizes = [2**i for i in range(0, 21)]
    
    results = {}
    
    for size in tqdm(message_sizes, desc="Running experiments"):
        print(f"\nTesting message size: {size} bytes")
        metrics = run_experiment(size, num_trials=100)
        results[size] = metrics
        
        print(f"  Round-trip latency: {metrics['round_trip_ns']['mean']:.2f} ns")
        print(f"  Throughput: {metrics['throughput_gbps']:.2f} GB/s")
    
    # Save results
    with open('warp_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to warp_results.json")

if __name__ == "__main__":
    main()