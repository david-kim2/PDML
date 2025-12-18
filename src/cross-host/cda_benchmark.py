import argparse
import os


def compute_max_args(min_host_mem, max_host_mem, min_gpu_mem, max_gpu_mem, fix_num_pairs):
    possible_pairs = [fix_num_pairs] if fix_num_pairs >= 1 else [1 << i for i in range(0, 11)]  # 1 - 1024
    possible_msg_sizes = [1 << i for i in range(0, 40)]  # 1B to 1TiB

    valid_configs = []
    for num_pairs in possible_pairs:
        for msg_size in possible_msg_sizes:
            host_mem_req = 2 * msg_size >= min_host_mem and 2 * msg_size <= max_host_mem
            gpu_mem_req = msg_size >= min_gpu_mem and msg_size <= max_gpu_mem
            if host_mem_req and gpu_mem_req and msg_size % num_pairs == 0:
                valid_configs.append((num_pairs, msg_size, 2 * msg_size, msg_size))

    return valid_configs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-host-mem", type=int, default=0, help="Minimum host memory in MiB")
    parser.add_argument("--max-host-mem", type=int, required=True, help="Maximum host memory in MiB")
    parser.add_argument("--min-gpu-mem", type=int, default=0, help="Minimum GPU memory in MiB")
    parser.add_argument("--max-gpu-mem", type=int, required=True, help="Maximum GPU memory in MiB")
    parser.add_argument("--fix-num-pairs", type=int, default=1, help="Fix number of thread pairs to this value")
    parser.add_argument("--num-runs", type=int, default=25, help="Number of runs per configuration")
    args = parser.parse_args()

    valid_configs = compute_max_args(
        args.min_host_mem * 1024 * 1024, args.max_host_mem * 1024 * 1024,
        args.min_gpu_mem * 1024 * 1024, args.max_gpu_mem * 1024 * 1024, 
        args.fix_num_pairs
    )
    for num_pairs, msg_size, host_mem, gpu_mem in valid_configs:
        cmd = f"./bin_cda_main {msg_size} {num_pairs} {args.num_runs}"
        print(f"Running with num_pairs={num_pairs}, msg_size={msg_size}, host_mem={host_mem} bytes, gpu_mem={gpu_mem} bytes")
        os.system(cmd)
        print("============================================")
