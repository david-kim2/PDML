#include "common.hpp"
#include <stdint.h>
#include <cstdio>
#include <cuda.h>


__device__ __forceinline__ uint32_t get_timestamp() {
    uint32_t ret;
    asm volatile("mov.u32 %0, %globaltimer_lo;" : "=r"(ret));
    return ret;
}


__global__ void cross_warp_echo_kernel(
    uint32_t* metrics, // Place to store output metrics, shape [n_runs][num_pairs][6]
    int msg_size, // Message size in bytes, split evenly between pairs
    int num_pairs, // Number of client-server pairs (threads per side)
    int n_runs // Number of runs to perform to allow averaging
) {
    // Thread variables
    int tid             = threadIdx.x;
    int warp_id         = tid / 32; // 0 = clients, 1 = servers
    int lane_id         = tid % 32; // index within warp
    int msg_size_thread = msg_size / num_pairs;

    // Shared memory layout, size = (num_pairs*2 * msg_size_thread) + (num_pairs*2) bytes
    extern __shared__ uint8_t shm[];
    uint8_t* client_offset         = shm + (lane_id*2) * msg_size_thread;
    uint8_t* server_offset         = client_offset + msg_size_thread;
    volatile uint8_t* finished_c2s = (volatile uint8_t*)(shm + (num_pairs*2) * msg_size_thread);
    volatile uint8_t* finished_s2c = (volatile uint8_t*)(finished_c2s + num_pairs);

    // Iterating over different runs
    for (int run = 0; run < n_runs; run++) {
        int output_idx = (run * num_pairs + lane_id) * 6;
        unsigned mask = __ballot_sync(0xFFFFFFFF, lane_id < num_pairs);

        if (warp_id == 0 && lane_id < num_pairs) { // CLIENT
            // Reset client-to-server buffer to zero
            for (int i = 0; i < msg_size_thread; i++)
                client_offset[i] = 0;
            finished_c2s[lane_id] = 0;
            __syncwarp(mask);

            // Begin client-to-server communication
            uint32_t client_start, client_end, client_recv;
            client_start = get_timestamp();

            for (int i = 0; i < msg_size_thread; i++)
                client_offset[i] = uint8_t(lane_id + 1);

            __threadfence_block();
            finished_c2s[lane_id] = 1;
            client_end = get_timestamp();

            // Wait for server response
            while (finished_s2c[lane_id] == 0);
            client_recv = get_timestamp();

            // Confirm data integrity
            bool valid = true;
            for (int i = 0; i < msg_size_thread; i++)
                valid &= (server_offset[i] == uint8_t(lane_id + 1));

            // Store client metrics
            if (!valid) {
                metrics[output_idx + 0] = 0xFFFFFFFF;
                metrics[output_idx + 1] = 0xFFFFFFFF;
                metrics[output_idx + 2] = 0xFFFFFFFF;
            } else {
                metrics[output_idx + 0] = client_start;
                metrics[output_idx + 1] = client_end;
                metrics[output_idx + 2] = client_recv;
            }
        } else if (warp_id == 1 && lane_id < num_pairs) { // SERVER
            // Reset server-to-client buffer to zero
            for (int i = 0; i < msg_size_thread; i++)
                server_offset[i] = 0;
            finished_s2c[lane_id] = 0;
            __syncwarp(mask);

            // Wait for client response
            uint32_t server_recv, server_start, server_end;
            while (finished_c2s[lane_id] == 0);
            server_recv = get_timestamp();

            // Begin client-to-server communication
            server_start = get_timestamp();
            for (int i = 0; i < msg_size_thread; i++)
                server_offset[i] = client_offset[i];

            __threadfence_block();
            finished_s2c[lane_id] = 1;
            server_end = get_timestamp();

            // Store server metrics
            metrics[output_idx + 3] = server_recv;
            metrics[output_idx + 4] = server_start;
            metrics[output_idx + 5] = server_end;
        }

        __syncthreads();
    }
}
