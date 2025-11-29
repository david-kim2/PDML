#include "../common.hpp"
#include <stdint.h>
#include <cstdio>
#include <cuda.h>


__device__ __forceinline__ uint64_t get_timestamp() {
    uint64_t ret;
    asm volatile("mov.u64 %0, %globaltimer;" : "=l"(ret));
    return ret;
}


__global__ void cross_thread_echo_kernel(
    uint64_t* metrics, // Place to store output metrics, shape [n_runs][num_pairs][8]
    int msg_size, // Message size in bytes, split evenly between pairs
    int num_pairs, // Number of client-server pairs (threads per side)
    int n_runs // Number of runs to perform to allow averaging
) {
    // Thread variables
    int tid             = threadIdx.x;
    int msg_size_thread = msg_size / num_pairs;
    int msg_size_word   = (msg_size_thread + 3) / 4;
    unsigned mask       = __ballot_sync(0xFFFFFFFF, tid < 2 * num_pairs);

    int pair       = tid % num_pairs;
    bool is_client = (tid < num_pairs);
    bool is_server = (tid >= num_pairs && tid < 2 * num_pairs);

    // Iterating over different runs
    for (int run = 0; run < n_runs; run++) {
        int output_idx = (run * num_pairs + pair) * 8;

        if (tid < 2 * num_pairs) {
            // Begin client-to-server communication
            uint64_t start = 0, end = 0, recv_start = 0, recv_end = 0;
            uint32_t expected    = uint32_t(pair + 1);
            uint32_t client_send = is_client ? expected : 0u;

            bool server_valid        = true;
            if (is_client) start     = get_timestamp();
            uint32_t val_from_client = __shfl_up_sync(mask, client_send, num_pairs);

            if (is_server) {
                recv_start    = get_timestamp();
                server_valid &= (val_from_client == expected);
            }

            for (int i = 1; i < msg_size_word; i++) {
                val_from_client = __shfl_up_sync(mask, client_send, num_pairs);
                server_valid &= (val_from_client == expected);
            }

            if (is_client) end      = get_timestamp();
            if (is_server) recv_end = get_timestamp();
            __syncwarp(mask);

            // Begin server-to-client communication
            uint32_t server_send = is_server && server_valid ? expected : 0u;

            bool client_valid        = true;
            if (is_server) start     = get_timestamp();
            uint32_t val_from_server = __shfl_down_sync(mask, server_send, num_pairs);

            if (is_client) {
                recv_start    = get_timestamp();
                client_valid &= (val_from_server == expected);
            }

            for (int i = 1; i < msg_size_word; i++) {
                val_from_server = __shfl_down_sync(mask, server_send, num_pairs);
                client_valid &= (val_from_server == expected);
            }

            if (is_server) end      = get_timestamp();
            if (is_client) recv_end = get_timestamp();
            __syncwarp(mask);

            // Store metrics
            int offset = is_client ? 0 : 4;
            if ((is_client && client_valid) || (is_server && server_valid)) {
                metrics[output_idx + offset + 0] = start;
                metrics[output_idx + offset + 1] = end;
                metrics[output_idx + offset + 2] = recv_start;
                metrics[output_idx + offset + 3] = recv_end;
            } else {
                metrics[output_idx + offset + 0] = 0ull;
                metrics[output_idx + offset + 1] = 0ull;
                metrics[output_idx + offset + 2] = 0ull;
                metrics[output_idx + offset + 3] = 0ull;
            }
        }

        __syncthreads();
    }
}
