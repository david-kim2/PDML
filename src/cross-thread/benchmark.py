import argparse
import os


def compute_max_args(max_mem, fix_num_pairs):
    possible_pairs = [fix_num_pairs] if fix_num_pairs >= 1 else [1 << i for i in range(0, 5)]  # 1,2,4,8,16
    possible_msg_sizes = [1 << i for i in range(0, 30)]  # 1B to 1GiB

    valid_configs = []
    for num_pairs in possible_pairs:
        for msg_size in possible_msg_sizes:
            if msg_size <= max_mem and msg_size % num_pairs == 0:
                valid_configs.append((num_pairs, msg_size))

    return valid_configs


if __name__ == "__main__":
    # We need to pass in max shared memory per block
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-mem", type=int, required=True, help="Maximum shared memory per block in KiB")
    parser.add_argument("--fix-num-pairs", type=int, default=1, help="Fix number of thread pairs to this value (<1 for auto)")
    parser.add_argument("--num-runs", type=int, default=10, help="Number of runs per configuration")
    args = parser.parse_args()

    valid_configs = compute_max_args(args.max_mem * 1024, args.fix_num_pairs)
    for num_pairs, msg_size in valid_configs:
        cmd = f"./bin_main {msg_size} {num_pairs} {args.num_runs}"
        print(f"Running with num_pairs={num_pairs}, msg_size={msg_size}")
        os.system(cmd)
        print("============================================")
