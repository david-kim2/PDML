import argparse
import os
import sys


def compute_valid_configs(fix_num_pairs):
    possible_pairs = [fix_num_pairs] if fix_num_pairs >= 1 else [1, 2, 4, 8, 16, 32]
    possible_msg_sizes = [1 << i for i in range(0, 31)]
    
    valid_configs = []
    for num_pairs in possible_pairs:
        for msg_size in possible_msg_sizes:
            if msg_size % num_pairs == 0:
                valid_configs.append((num_pairs, msg_size))
    
    return valid_configs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Cross-Node Echo Benchmark')
    parser.add_argument("--max-shared-mem", type=int, default=0, 
                       help="Not used for cross-node (kept for compatibility)")
    parser.add_argument("--fix-num-pairs", type=int, default=1, 
                       help="Fix number of thread pairs to this value (<1 for auto)")
    parser.add_argument("--num-runs", type=int, default=10, 
                       help="Number of runs per configuration")
    parser.add_argument("--nodes", type=int, default=2,
                       help="Number of nodes (must be 2)")
    parser.add_argument("--max-size", type=int, default=26,
                       help="Maximum message size as power of 2 (default: 26 = 64MB)")
    args = parser.parse_args()
    
    if args.nodes != 2:
        print("Error: This benchmark requires exactly 2 nodes")
        sys.exit(1)
    
    # Set NVSHMEM bootstrap for SLURM/PMI
    os.environ['NVSHMEM_BOOTSTRAP'] = 'PMI'
    os.environ['NVSHMEM_DISABLE_CUDA_VMM'] = '1'
    
    # Check if we're already in an srun context
    if 'SLURM_JOB_ID' not in os.environ:
        print("Warning: Not running under SLURM. Make sure you have an allocation.")
        print("Recommended: salloc -N 2 --qos interactive --time 01:00:00 --constraint gpu --gpus-per-node=1")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(0)
    
    valid_configs = compute_valid_configs(args.fix_num_pairs)
    valid_configs = [(pairs, size) for pairs, size in valid_configs 
                     if size <= (1 << args.max_size)]
    
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
        
        cmd = f"srun -N 2 -n 2 --gpus-per-task=1 --gpu-bind=none ./bin_main {msg_size} {num_pairs} {args.num_runs}"
        
        print(f"Running: num_pairs={num_pairs}, msg_size={size_str} ({msg_size} bytes)")
        print(f"Command: {cmd}")
        
        ret = os.system(cmd)
        
        if ret != 0:
            print(f"Error: Command failed with return code {ret}")
            print("Stopping benchmark.")
            sys.exit(1)
        
        print("=" * 60)
    
    print("\nBenchmark completed")
