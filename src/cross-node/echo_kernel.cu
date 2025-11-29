#include <nvshmem.h>
#include <nvshmemx.h>
#include <stdint.h>
#include <cstdio>


__device__ __forceinline__ uint64_t get_timestamp() {
    uint64_t ret;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(ret));
    return ret;
}


__global__ void cross_node_echo_kernel(
    uint64_t* metrics,
    uint8_t* nvshmem_buffer,
    int msg_size,
    int num_pairs,
    int n_runs,
    int my_pe,
    int peer_pe,
    uint64_t clock_offset
) {
    int tid = threadIdx.x;
    int lane_id = tid % 32;
    
    if (lane_id >= num_pairs) return;
    
    int msg_size_thread = msg_size / num_pairs;
    
    uint8_t* my_msg_buffer = nvshmem_buffer + lane_id * msg_size_thread;
    volatile uint8_t* my_flag = (volatile uint8_t*)(nvshmem_buffer + num_pairs * msg_size_thread + lane_id);
    
    uint8_t* peer_msg_buffer = my_msg_buffer;
    volatile uint8_t* peer_flag = my_flag;
    
    for (int run = 0; run < n_runs; run++) {
        int output_idx = (run * num_pairs + lane_id) * 8;
        
        *my_flag = 0;
        __threadfence_system();
        
        if (my_pe == 0) {
            uint64_t client_start, client_end, client_recv_start, client_recv_end;
            
            uint8_t fill = uint8_t(lane_id + 1);
            for (int i = 0; i < msg_size_thread; i++) {
                my_msg_buffer[i] = fill;
            }
            __threadfence_system();
            
            client_start = get_timestamp();
            
            // Use nvshmem_putmem_nbi (thread-level, not nvshmemx)
            nvshmem_putmem_nbi(peer_msg_buffer, my_msg_buffer, msg_size_thread, peer_pe);
            
            uint8_t flag_val = 1;
            nvshmem_uint8_p((uint8_t*)peer_flag, flag_val, peer_pe);
            
            nvshmem_quiet();
            
            client_end = get_timestamp();
            
            while (*my_flag != 2) {}
            client_recv_start = get_timestamp();
            
            bool valid = true;
            for (int i = 0; i < msg_size_thread; i++) {
                valid &= (my_msg_buffer[i] == fill);
            }
            
            client_recv_end = get_timestamp();
            
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
            
        } else {
            uint64_t server_recv_start, server_recv_end, server_start, server_end;
            
            while (*my_flag != 1) {}
            server_recv_start = get_timestamp();
            
            // Actually read the entire message to get measurable receive time
            uint8_t checksum = 0;
            for (int i = 0; i < msg_size_thread; i++) {
                checksum ^= my_msg_buffer[i];
            }
            
            server_recv_end = get_timestamp();
            
            server_start = get_timestamp();
            
            nvshmem_putmem_nbi(peer_msg_buffer, my_msg_buffer, msg_size_thread, peer_pe);
            
            uint8_t flag_val = 2;
            nvshmem_uint8_p((uint8_t*)peer_flag, flag_val, peer_pe);
            
            nvshmem_quiet();
            
            server_end = get_timestamp();
            
            // Server (PE 1) writes its timestamps directly to PE 0's metrics array
            // Adjust timestamps with clock offset
            uint64_t ts0 = server_recv_start - clock_offset;
            uint64_t ts1 = server_recv_end - clock_offset;
            uint64_t ts2 = server_start - clock_offset;
            uint64_t ts3 = server_end - clock_offset;
            
            nvshmem_uint64_p(metrics + output_idx + 4, ts0, peer_pe);
            nvshmem_uint64_p(metrics + output_idx + 5, ts1, peer_pe);
            nvshmem_uint64_p(metrics + output_idx + 6, ts2, peer_pe);
            nvshmem_uint64_p(metrics + output_idx + 7, ts3, peer_pe);
            nvshmem_quiet();
        }
        
        // Barrier to ensure server's timestamp writes complete before next iteration
        nvshmem_barrier_all();
    }
}


__global__ void ping_pong_kernel(
    uint64_t* timestamps,
    uint8_t* ping_buffer,
    int my_pe,
    int peer_pe
) {
    if (threadIdx.x != 0) return;
    
    volatile uint8_t* my_flag = (volatile uint8_t*)ping_buffer;
    volatile uint8_t* peer_flag = my_flag;
    
    *my_flag = 0;
    __threadfence_system();
    nvshmem_barrier_all();
    
    if (my_pe == 0) {
        timestamps[0] = get_timestamp();
        uint8_t val = 1;
        nvshmem_uint8_p((uint8_t*)peer_flag, val, peer_pe);
        nvshmem_quiet();
        
        while (*my_flag != 2) {}
        timestamps[1] = get_timestamp();
        
    } else {
        while (*my_flag != 1) {}
        timestamps[2] = get_timestamp();
        
        timestamps[3] = get_timestamp();
        uint8_t val = 2;
        nvshmem_uint8_p((uint8_t*)peer_flag, val, peer_pe);
        nvshmem_quiet();
    }
    
    nvshmem_barrier_all();
}
