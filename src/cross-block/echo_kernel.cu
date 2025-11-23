#include <cooperative_groups.h>
#include "../common.hpp"
#include <stdint.h>
#include <cstdio>
#include <cuda.h>


namespace cg = cooperative_groups;

__device__ __forceinline__ uint64_t get_timestamp() {
    uint64_t ret;
    asm volatile("mov.u64 %0, %globaltimer;" : "=l"(ret));
    return ret;
}


__global__ void cross_block_echo_kernel(
    uint64_t* metrics, // Place to store output metrics, shape [n_runs][num_pairs][8]
    uint8_t* client_buf, // Global memory buffer for client-to-server messages
    uint8_t* server_buf, // Global memory buffer for server-to-client messages
    volatile uint8_t* finished_c2s, // Global memory flags for client-to-server completion
    volatile uint8_t* finished_s2c, // Global memory flags for server-to-client completion
    int msg_size, // Message size in bytes, split evenly between pairs
    int num_pairs, // Number of client-server pairs (threads per side)
    int n_runs // Number of runs to perform to allow averaging
) {
    // Thread variables
    cg::grid_group grid = cg::this_grid();
    int bid             = blockIdx.x;
    int tid             = threadIdx.x;
    int msg_size_thread = msg_size / num_pairs;

    // Global memory layout, size = (num_pairs * msg_size_thread) + (num_pairs)
    uint8_t* client_offset = client_buf + tid * msg_size_thread;
    uint8_t* server_offset = server_buf + tid * msg_size_thread;

    // Iterating over different runs
    for (int run = 0; run < n_runs; run++) {
        int output_idx = (run * num_pairs + tid) * 8;

        if (bid == 0 && tid < num_pairs) { // CLIENT
            // Reset client-to-server buffer to zero
            for (int i = 0; i < msg_size_thread; i++)
                client_offset[i] = 0;
            finished_c2s[tid] = 0;
            __threadfence();
            grid.sync();

            // Begin client-to-server communication
            uint64_t client_start, client_end, client_recv_start, client_recv_end;
            client_start = get_timestamp();
            finished_c2s[tid] = 1;
            __threadfence();

            uint8_t fill = uint8_t(tid + 1);
            for (int i = 0; i < msg_size_thread; i++)
                client_offset[i] = fill;
            __threadfence();

            finished_c2s[tid] = 2;
            __threadfence();
            client_end = get_timestamp();

            // Wait for server response
            while (finished_s2c[tid] != 1);
            client_recv_start = get_timestamp();
            while (finished_s2c[tid] != 2);
            client_recv_end = get_timestamp();

            // Confirm data integrity
            bool valid = true;
            for (int i = 0; i < msg_size_thread; i++)
                valid &= (server_offset[i] == fill);

            // Store client metrics
            if (!valid) {
                metrics[output_idx + 0] = 0ull;
                metrics[output_idx + 1] = 0ull;
                metrics[output_idx + 2] = 0ull;
                metrics[output_idx + 3] = 0ull;
            } else {
                metrics[output_idx + 0] = client_start;
                metrics[output_idx + 1] = client_end;
                metrics[output_idx + 2] = client_recv_start;
                metrics[output_idx + 3] = client_recv_end;
            }
        } else if (bid == 1 && tid < num_pairs) { // SERVER
            // Reset server-to-client buffer to zero
            for (int i = 0; i < msg_size_thread; i++)
                server_offset[i] = 0;
            finished_s2c[tid] = 0;
            __threadfence();
            grid.sync();

            // Wait for client response
            uint64_t server_recv_start, server_recv_end, server_start, server_end;
            while (finished_c2s[tid] != 1);
            server_recv_start = get_timestamp();
            while (finished_c2s[tid] != 2);
            server_recv_end = get_timestamp();

            // Begin client-to-server communication
            server_start = get_timestamp();
            finished_s2c[tid] = 1;
            __threadfence();

            for (int i = 0; i < msg_size_thread; i++)
                server_offset[i] = client_offset[i];
            __threadfence();

            finished_s2c[tid] = 2;
            __threadfence();
            server_end = get_timestamp();

            // Store server metrics
            metrics[output_idx + 4] = server_recv_start;
            metrics[output_idx + 5] = server_recv_end;
            metrics[output_idx + 6] = server_start;
            metrics[output_idx + 7] = server_end;
        }

        __syncthreads();
    }
}
