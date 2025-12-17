import argparse
import os


def compute_max_args(max_shared_mem, fix_num_pairs):
    possible_pairs = [fix_num_pairs] if fix_num_pairs >= 1 else [1 << i for i in range(0, 10)]  # 1 - 512
    possible_msg_sizes = [1 << i for i in range(0, 40)]  # 1B to 1TiB

    valid_configs = []
    for num_pairs in possible_pairs:
        for msg_size in possible_msg_sizes:
            shared_mem_per_block = (2 * msg_size) + (2 * num_pairs)
            if shared_mem_per_block <= max_shared_mem and msg_size % num_pairs == 0:
                valid_configs.append((num_pairs, msg_size, shared_mem_per_block))

    return valid_configs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-shared-mem", type=int, required=True, help="Maximum shared memory per block in KiB")
    parser.add_argument("--fix-num-pairs", type=int, default=1, help="Fix number of thread pairs to this value")
    parser.add_argument("--num-runs", type=int, default=10, help="Number of runs per configuration")
    args = parser.parse_args()

    valid_configs = compute_max_args(args.max_shared_mem * 1024, args.fix_num_pairs)
    for num_pairs, msg_size, shared_mem in valid_configs:
        cmd = f"./bin_main {msg_size} {num_pairs} {args.num_runs}"
        print(f"Running with num_pairs={num_pairs}, msg_size={msg_size}, shared_mem={shared_mem} bytes")
        os.system(cmd)
        print("============================================")
