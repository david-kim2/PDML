import argparse
import os

launcher_path = "launcher.cu"

def compute_max_args(max_shared_mem, fix_num_pairs):
    possible_pairs = [fix_num_pairs] if fix_num_pairs >= 1 else range(1, 32)
    possible_msg_sizes = [1 << i for i in range(0, 20)]  # 1B to 1MiB

    valid_configs = []
    for num_pairs in possible_pairs:
        for msg_size in possible_msg_sizes:
            # shared_mem_per_block = num_pairs * msg_size * 2 + num_pairs * 8 * 4  # data + control
            # int msg_size_thread    = msg_size / num_pairs;
            # size_t shared_mem_size = (2 * num_pairs * msg_size_thread) + (2 * num_pairs);
            msg_size_thread    = msg_size // num_pairs
            shared_mem_per_block = (2 * num_pairs * msg_size_thread) + (2 * num_pairs)
            if shared_mem_per_block <= max_shared_mem:
                valid_configs.append((num_pairs, msg_size, shared_mem_per_block))
    

if __name__ == "__main__":
    # we need to pass in max shared memory per block
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-shared-mem", type=int, required=True, help="Maximum shared memory per block in bytes")
    parser.add_argument("--fix-num-pairs", type=int, default=1, help="Fix number of warp pairs to this value (<1 for auto)")
    args = parser.parse_args()

