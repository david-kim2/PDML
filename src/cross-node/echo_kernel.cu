#include <nvshmem.h>
#include <nvshmemx.h>
#include <stdint.h>
#include <cstdio>


__device__ __forceinline__ uint64_t get_timestamp()
{
    uint64_t ret;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(ret));
    return ret;
}


__global__ void cross_node_echo_kernel(
    uint64_t *metrics,
    uint8_t *nvshmem_buffer,
    int msg_size,
    int num_pairs,
    int n_runs,
    int my_pe,
    int peer_pe,
    int64_t clock_offset  // Changed from uint64_t to int64_t
)
{
    int tid = threadIdx.x;
    
    if (tid >= num_pairs) return;
    
    int msg_size_thread = msg_size / num_pairs;
    
    uint8_t* my_msg_buffer = nvshmem_buffer + tid * msg_size_thread;
    volatile uint8_t* my_flag = (volatile uint8_t*)(nvshmem_buffer + num_pairs * msg_size_thread + tid);
    
    uint8_t* peer_msg_buffer = my_msg_buffer;
    volatile uint8_t* peer_flag = my_flag;
    
    for (int run = 0; run < n_runs; ++run)
    {
        int output_idx = (run * num_pairs + tid) * 8;
        
        *my_flag = 0;
        __threadfence_system();
        __syncthreads();
        if (tid == 0)
        {
            nvshmem_barrier_all();
        }
        __syncthreads();
        
        if (my_pe == 0)
        {
            // CLIENT
            uint64_t client_trans_start, client_trans_end, client_recv_start, client_recv_end;
            
            uint8_t fill = uint8_t(tid + 1);
            for (int i = 0; i < msg_size_thread; ++i)
            {
                my_msg_buffer[i] = fill;
            }
            __threadfence_system();
            
            client_trans_start = get_timestamp();
            __threadfence();
            
            nvshmem_putmem_nbi(peer_msg_buffer, my_msg_buffer, msg_size_thread, peer_pe);
            nvshmem_uint8_p((uint8_t*)peer_flag, (uint8_t)1, peer_pe);
            nvshmem_quiet();
            
            client_trans_end = get_timestamp();
            __threadfence();

            // Sync PE0 threads
            __syncthreads();

            // Cross-PE barrier
            if (tid == 0) {
                nvshmem_barrier_all();
            }
            __syncthreads();

            // wait for server to echo back
            while (*my_flag != 2) {}
            
            client_recv_start = get_timestamp();
            __threadfence();
            
            bool valid = true;
            for (int i = 0; i < msg_size_thread; ++i)
            {
                valid &= (my_msg_buffer[i] == fill);
            }
            
            client_recv_end = get_timestamp();
            __threadfence();
            
            if (!valid)
            {
                metrics[output_idx + 0] = 0ull;
                metrics[output_idx + 1] = 0ull;
                metrics[output_idx + 2] = 0ull;
                metrics[output_idx + 3] = 0ull;
            }
            else
            {
                metrics[output_idx + 0] = client_trans_start;
                metrics[output_idx + 1] = client_trans_end;
                metrics[output_idx + 2] = client_recv_start;
                metrics[output_idx + 3] = client_recv_end;
            }
            
        }
        else
        {
            // SERVER
            uint64_t server_recv_start, server_recv_end, server_trans_start, server_trans_end;
            
            while (*my_flag != 1) {}
            
            server_recv_start = get_timestamp();
            __threadfence();
            
            uint8_t checksum = 0;
            for (int i = 0; i < msg_size_thread; ++i)
            {
                checksum ^= my_msg_buffer[i];
            }
            
            server_recv_end = get_timestamp();
            __threadfence();

            // Sync all threads within PE1 first
            __threadfence_system();
            __syncthreads();

            // NOW sync across PEs before anyone transmits
            if (tid == 0) {
                nvshmem_barrier_all();  // ← This ensures PE0 and PE1 are both done receiving
            }
            __syncthreads();

            server_trans_start = get_timestamp();
            __threadfence();
            
            nvshmem_putmem_nbi(peer_msg_buffer, my_msg_buffer, msg_size_thread, peer_pe);
            nvshmem_uint8_p((uint8_t*)peer_flag, (uint8_t)2, peer_pe);
            nvshmem_quiet();
            
            server_trans_end = get_timestamp();
            __threadfence();
            
            // Adjust server timestamps to PE0's clock domain using signed offset
            // offset = (PE1_clock - PE0_clock), so subtract to convert PE1 -> PE0 time
            uint64_t ts0 = (uint64_t)((int64_t)server_recv_start - clock_offset);
            uint64_t ts1 = (uint64_t)((int64_t)server_recv_end - clock_offset);
            uint64_t ts2 = (uint64_t)((int64_t)server_trans_start - clock_offset);
            uint64_t ts3 = (uint64_t)((int64_t)server_trans_end - clock_offset);
            
            nvshmem_uint64_p(metrics + output_idx + 4, ts0, peer_pe);
            nvshmem_uint64_p(metrics + output_idx + 5, ts1, peer_pe);
            nvshmem_uint64_p(metrics + output_idx + 6, ts2, peer_pe);
            nvshmem_uint64_p(metrics + output_idx + 7, ts3, peer_pe);
            nvshmem_quiet();
        }
        
        __syncthreads();
        if (tid == 0)
        {
            nvshmem_barrier_all();
        }
        __syncthreads();
    }
}


__global__ void ping_pong_kernel(
    uint64_t* timestamps,
    uint8_t* ping_buffer,
    int my_pe,
    int peer_pe
)
{
    if (threadIdx.x != 0) return;
    
    volatile uint8_t* my_flag = (volatile uint8_t*)ping_buffer;
    volatile uint8_t* peer_flag = my_flag;
    
    *my_flag = 0;
    __threadfence_system();
    nvshmem_barrier_all();
    
    if (my_pe == 0)
    {
        uint64_t t0 = get_timestamp();
        __threadfence();
        
        // Send signal to PE1 immediately after timestamp
        uint8_t val = 1;
        nvshmem_uint8_p((uint8_t*)peer_flag, val, peer_pe);
        nvshmem_quiet();
        
        // Wait for response
        while (*my_flag != 2) {}
        
        uint64_t t3 = get_timestamp();
        __threadfence();
        
        timestamps[0] = t0;
        timestamps[3] = t3;
        
    }
    else
    {
        // Wait for signal from PE0
        while (*my_flag != 1) {}
        
        uint64_t t1 = get_timestamp();
        __threadfence();
        
        uint64_t t2 = get_timestamp();
        __threadfence();
        
        // Send response immediately after timestamps
        uint8_t val = 2;
        nvshmem_uint8_p((uint8_t*)peer_flag, val, peer_pe);
        nvshmem_quiet();
        
        nvshmem_uint64_p(timestamps + 1, t1, peer_pe);
        nvshmem_uint64_p(timestamps + 2, t2, peer_pe);
        nvshmem_quiet();
    }
    
    nvshmem_barrier_all();
}
