import argparse
import os


def compute_max_args(min_mem, max_mem, fix_num_pairs):
    possible_pairs = [fix_num_pairs] if fix_num_pairs >= 1 else [1 << i for i in range(0, 6)]  # 1,2,4,8,16,32
    possible_msg_sizes = [1 << i for i in range(0, 40)]  # 1B to 1TiB

    valid_configs = []
    for num_pairs in possible_pairs:
        for msg_size in possible_msg_sizes:
            msg_size_thread = msg_size // num_pairs
            buf_size = num_pairs * msg_size_thread
            flag_size = num_pairs
            global_mem_size = 2 * buf_size + 2 * flag_size
            if global_mem_size >= min_mem and global_mem_size <= max_mem and msg_size % num_pairs == 0:
                valid_configs.append((num_pairs, msg_size, global_mem_size))

    return valid_configs


if __name__ == "__main__":
    # We need to pass in max shared memory per block
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-mem", type=int, default=0, help="Minimum shared memory per block in MiB")
    parser.add_argument("--max-mem", type=int, required=True, help="Maximum shared memory per block in MiB")
    parser.add_argument("--fix-num-pairs", type=int, default=1, help="Fix number of thread pairs to this value (<1 for auto)")
    parser.add_argument("--num-runs", type=int, default=10, help="Number of runs per configuration")
    args = parser.parse_args()

    valid_configs = compute_max_args(args.max_mem * 1024 * 1024, args.fix_num_pairs)
    for num_pairs, msg_size, mem in valid_configs:
        cmd = f"./bin_main {msg_size} {num_pairs} {args.num_runs}"
        print(f"Running with num_pairs={num_pairs}, msg_size={msg_size}, mem={mem} bytes")
        os.system(cmd)
        print("============================================")
