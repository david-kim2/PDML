import argparse
import os
import sys


def compute_valid_configs(fix_num_pairs, fix_msg_size, max_msg_size):
    possible_pairs = [fix_num_pairs] if fix_num_pairs >= 1 else [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    possible_msg_sizes = [1 << fix_msg_size] if fix_msg_size >= 0 else [1 << i for i in range(0, max_msg_size + 1)]
    
    valid_configs = []
    for num_pairs in possible_pairs:
        for msg_size in possible_msg_sizes:
            if msg_size % num_pairs == 0:
                valid_configs.append((num_pairs, msg_size))
    
    return valid_configs


if __name__ == "__main__":
    # salloc --nodes 1 --qos interactive --time 03:00:00 --constraint gpu --gpus 2 --account m4999
    parser = argparse.ArgumentParser(description='Cross-Node Echo Benchmark')
    parser.add_argument("--fix-num-pairs", type=int, default=1, 
                       help="Fix number of thread pairs to this value (<1 for auto)")
    parser.add_argument("--num-runs", type=int, default=10, 
                       help="Number of runs per configuration")
    parser.add_argument("--max-size", type=int, default=26,
                       help="Maximum message size as power of 2 (default: 26 = 64MB)")
    parser.add_argument("--fix-msg-size", type=int, default=-1,
                       help="Fix message size to this value in bytes (<0 for auto)")
    args = parser.parse_args()
    
    valid_configs = compute_valid_configs(args.fix_num_pairs, args.fix_msg_size, args.max_size)
    
    # print(valid_configs)
    for num_pairs, msg_size in valid_configs:
        # Format message size for display
        if msg_size >= 1024*1024*1024:
            size_str = f"{msg_size // (1024*1024*1024)}GB"
        elif msg_size >= 1024*1024:
            size_str = f"{msg_size // (1024*1024)}MB"
        elif msg_size >= 1024:
            size_str = f"{msg_size // 1024}KB"
        else:
            size_str = f"{msg_size}B"
        
        cmd = f"MPICH_GPU_SUPPORT_ENABLED=0 srun -n 2 -G 2 --mpi=pmi2 ./bin_main {msg_size} {num_pairs} {args.num_runs}"
        
        print(f"Running: num_pairs={num_pairs}, msg_size={size_str} ({msg_size} bytes)")
        print(f"Command: {cmd}")
        
        ret = os.system(cmd)
        
        if ret != 0:
            print(f"Error: Command failed with return code {ret}")
            print("Stopping benchmark.")
            sys.exit(1)
        
        print("=" * 60)
    
    print("\nBenchmark completed")
