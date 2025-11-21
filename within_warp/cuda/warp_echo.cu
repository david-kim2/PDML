#include <torch/extension.h>
#include <cuda_runtime.h>

__device__ __forceinline__ uint32_t get_timestamp() {
  volatile uint32_t ret;
  asm volatile("mov.u32 %0, %globaltimer_lo;" : "=r"(ret));
  return ret;
}

__global__ void warp_echo_kernel(
    uint32_t* results,      // cl_tx_st, cl_tx_ed, cl_rx, sv_rx, sv_tx_st, sv_tx_ed
    uint32_t* message,
    uint32_t num_words
) {
    int lane_id = threadIdx.x % 32;
    
    if (lane_id == 0) {
        // CLIENT
        results[0] = get_timestamp(); // cl_tx_st
        
        // Send message words via shuffle
        for (uint32_t i = 0; i < num_words; i++) {
            uint32_t data = message[i];
            // Shuffle to server (lane 1)
            __shfl_sync(0xffffffff, data, 1);
        }
        
        results[1] = get_timestamp(); // cl_tx_ed
        
        // Wait for response from server
        uint32_t response = __shfl_sync(0xffffffff, 0, 1);
        
        results[2] = get_timestamp(); // cl_rx
    }
    else if (lane_id == 1) {
        // SERVER
        // Receive message from client
        for (uint32_t i = 0; i < num_words; i++) {
            uint32_t data = __shfl_sync(0xffffffff, message[i], 0);
        }
        
        results[3] = get_timestamp(); // sv_rx
        results[4] = get_timestamp(); // sv_tx_st
        
        // Echo back
        for (uint32_t i = 0; i < num_words; i++) {
            uint32_t data = message[i];
            __shfl_sync(0xffffffff, data, 0);
        }
        
        results[5] = get_timestamp(); // sv_tx_ed
    }
}

torch::Tensor warp_echo_forward(
    torch::Tensor message,
    int message_size_bytes
) {
    auto results = torch::zeros({6}, torch::dtype(torch::kUInt32).device(message.device()));
    
    uint32_t num_words = (message_size_bytes + 3) / 4;
    
    // Launch kernel: 1 block, 32 threads
    warp_echo_kernel<<<1, 32>>>(
        results.data_ptr<uint32_t>(),
        message.data_ptr<uint32_t>(),
        num_words
    );
    
    cudaDeviceSynchronize();
    
    return results;
}

// PyBind11 bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("warp_echo_forward", &warp_echo_forward, "Warp echo benchmark");
}