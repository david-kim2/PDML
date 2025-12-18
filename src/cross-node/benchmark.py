import argparse
import os
import sys


def compute_valid_configs(fix_num_pairs, fix_msg_size, max_msg_size):
    possible_pairs = [fix_num_pairs] if fix_num_pairs >= 1 else [1, 2, 4, 8, 16, 32]
    possible_msg_sizes = [1 << fix_msg_size] if fix_msg_size >= 0 else [1 << i for i in range(0, max_msg_size + 1)]
    
    valid_configs = []
    for num_pairs in possible_pairs:
        for msg_size in possible_msg_sizes:
            if msg_size % num_pairs == 0:
                valid_configs.append((num_pairs, msg_size))
    
    return valid_configs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Cross-Node Echo Benchmark')
    parser.add_argument("--fix-num-pairs", type=int, default=1, 
                       help="Fix number of thread pairs to this value (<1 for auto)")
    parser.add_argument("--num-runs", type=int, default=10, 
                       help="Number of runs per configuration")
    parser.add_argument("--nodes", type=int, default=2,
                       help="Number of nodes (must be 2)")
    parser.add_argument("--max-size", type=int, default=26,
                       help="Maximum message size as power of 2 (default: 26 = 64MB)")
    parser.add_argument("--fix-msg-size", type=int, default=-1,
                       help="Fix message size to 2^N bytes (<0 for auto)")
    args = parser.parse_args()
    
    if args.nodes != 2:
        print("Error: This benchmark requires exactly 2 nodes")
        sys.exit(1)
    
    # Check if we're in a SLURM environment
    if 'SLURM_JOB_ID' not in os.environ:
        print("Warning: Not running under SLURM. Make sure you have an allocation.")
        print("Recommended: salloc -N 2 --qos interactive --time 01:00:00 --constraint gpu --gpus-per-node=1")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(0)
    
    valid_configs = compute_valid_configs(args.fix_num_pairs, args.fix_msg_size, args.max_size)
    
    print(f"Running {len(valid_configs)} configurations...")
    print(f"Message sizes: {min(c[1] for c in valid_configs)}B to {max(c[1] for c in valid_configs)}B")
    print(f"Thread pairs: {args.fix_num_pairs if args.fix_num_pairs >= 1 else 'variable (1-32)'}")
    print(f"Runs per config: {args.num_runs}")
    print()
    
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
        
        # Use PMI-2 bootstrap (same as your partner's cross-gpu implementation)
        cmd = f"MPICH_GPU_SUPPORT_ENABLED=0 srun -n 2 -G 2 --mpi=pmi2 ./bin_main {msg_size} {num_pairs} {args.num_runs}"
        
        print(f"Running: num_pairs={num_pairs}, msg_size={size_str} ({msg_size} bytes)")
        print(f"Command: {cmd}")
        
        ret = os.system(cmd)
        
        if ret != 0:
            print(f"Error: Command failed with return code {ret}")
            print("Stopping benchmark.")
            sys.exit(1)
        
        print("=" * 60)
    
    print("\nBenchmark completed successfully!")
