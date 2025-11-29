#include <nvshmem.h>
#include <nvshmemx.h>
#include <nlohmann/json.hpp>
#include <cuda_runtime.h>
#include <iostream>
#include <cassert>
#include <fstream>
#include <vector>
#include <algorithm>
#include <string>
#include <cstring>
#include <cstdlib>
#include <sys/stat.h>


__global__ void cross_node_echo_kernel(
    uint64_t* metrics,
    uint8_t* nvshmem_buffer,
    int msg_size,
    int num_pairs,
    int n_runs,
    int my_pe,
    int peer_pe,
    uint64_t clock_offset
);

__global__ void ping_pong_kernel(
    uint64_t* timestamps,
    uint8_t* ping_buffer,
    int my_pe,
    int peer_pe
);

#define TILE_SIZE 16

__global__ void warmup_kernel(float* A, float* B, float* C, int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; t++) {
        if (row < N && t * TILE_SIZE + threadIdx.x < N)
            As[threadIdx.y][threadIdx.x] = A[row * N + t * TILE_SIZE + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;
            
        if (col < N && t * TILE_SIZE + threadIdx.y < N)
            Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
            
        __syncthreads();
        
        for (int k = 0; k < TILE_SIZE; k++)
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
            
        __syncthreads();
    }
    
    if (row < N && col < N)
        C[row * N + col] = sum;
}


uint64_t calibrate_clocks(int my_pe, int peer_pe, uint8_t* ping_buffer) {
    const int num_calibrations = 10;
    std::vector<uint64_t> offsets;
    
    for (int i = 0; i < num_calibrations; i++) {
        uint64_t* d_timestamps;
        cudaMalloc(&d_timestamps, 4 * sizeof(uint64_t));
        cudaMemset(d_timestamps, 0, 4 * sizeof(uint64_t));
        
        ping_pong_kernel<<<1, 32>>>(d_timestamps, ping_buffer, my_pe, peer_pe);
        cudaDeviceSynchronize();
        
        uint64_t h_timestamps[4];
        cudaMemcpy(h_timestamps, d_timestamps, 4 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
        cudaFree(d_timestamps);
        
        if (my_pe == 0) {
            uint64_t pe0_send = h_timestamps[0];
            uint64_t pe0_recv = h_timestamps[1];
            uint64_t round_trip = pe0_recv - pe0_send;
            offsets.push_back(round_trip);
        }
    }
    
    if (my_pe == 0) {
        std::sort(offsets.begin(), offsets.end());
        uint64_t median_rtt = offsets[offsets.size() / 2];
        std::cout << "Median round-trip time: " << median_rtt << " cycles" << std::endl;
        std::cout << "Estimated one-way latency: " << median_rtt / 2 << " cycles" << std::endl;
        std::cout << "Note: Clock offset assumed to be 0 (same node)" << std::endl;
    }
    
    return 0;
}


void create_directories(const std::string& path) {
    size_t pos = 0;
    std::string dir;
    while ((pos = path.find('/', pos)) != std::string::npos) {
        dir = path.substr(0, pos++);
        if (!dir.empty()) {
            mkdir(dir.c_str(), 0755);
        }
    }
    mkdir(path.c_str(), 0755);
}


int main(int argc, char** argv) {
    // Set CUDA device based on local rank (before nvshmem_init)
    int local_rank = 0;
    char* local_rank_str = getenv("SLURM_LOCALID");
    if (local_rank_str) {
        local_rank = atoi(local_rank_str);
    }
    cudaSetDevice(local_rank);
    
    nvshmem_init();
    
    int my_pe = nvshmem_my_pe();
    int n_pes = nvshmem_n_pes();
    
    if (n_pes != 2) {
        if (my_pe == 0) {
            std::cerr << "Error: This program requires exactly 2 PEs (got " << n_pes << ")" << std::endl;
            std::cerr << "Run with: srun -N 2 --ntasks-per-node=1 --gpus-per-task=1 ./bin_main ..." << std::endl;
        }
        nvshmem_finalize();
        return 1;
    }
    
    int peer_pe = (my_pe == 0) ? 1 : 0;
    
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    
    if (my_pe == 0) {
        std::cout << "============================================" << std::endl;
        std::cout << "Cross-Node Echo Benchmark with NVSHMEM" << std::endl;
        std::cout << "============================================" << std::endl;
        std::cout << "Number of PEs: " << n_pes << std::endl;
        std::cout << "PE 0 Device: " << deviceProp.name << std::endl;
    }
    
    nvshmem_barrier_all();
    
    if (my_pe == 1) {
        std::cout << "PE 1 Device: " << deviceProp.name << std::endl;
    }
    
    nvshmem_barrier_all();
    
    if (my_pe == 0) {
        std::cout << "\nLaunching warmup kernel..." << std::endl;
    }
    
    int N = 8192;
    size_t bytes = N * N * sizeof(float);
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);
    
    dim3 blockDimWarmup(TILE_SIZE, TILE_SIZE);
    dim3 gridDimWarmup((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);
    warmup_kernel<<<gridDimWarmup, blockDimWarmup>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    if (my_pe == 0) {
        std::cout << "Warmup kernel completed." << std::endl;
    }
    
    nvshmem_barrier_all();
    
    int msg_size  = 1024;
    int num_pairs = 1;
    int n_runs    = 10;
    
    if (argc == 4) {
        msg_size  = std::stoi(argv[1]);
        num_pairs = std::stoi(argv[2]);
        n_runs    = std::stoi(argv[3]);
    } else if (my_pe == 0) {
        std::cout << "\nUsing default parameters:" << std::endl;
        std::cout << "Message Size: " << msg_size << " bytes" << std::endl;
        std::cout << "Number of Pairs: " << num_pairs << std::endl;
        std::cout << "Number of Runs: " << n_runs << std::endl;
    }
    
    assert(msg_size > 0 && "Message size must be greater than 0");
    assert((msg_size & (msg_size - 1)) == 0 && "Message size must be a power of 2");
    assert(num_pairs > 0 && "Number of pairs must be greater than 0");
    assert(num_pairs <= 32 && "Number of pairs must be less than or equal to 32");
    assert(msg_size % num_pairs == 0 && "Message size must be divisible by number of pairs");
    assert(n_runs > 0 && "Number of runs must be greater than 0");
    
    int msg_size_thread = msg_size / num_pairs;
    
    size_t buffer_size = (num_pairs * msg_size_thread) + num_pairs;
    uint8_t* nvshmem_buffer = (uint8_t*)nvshmem_malloc(buffer_size);
    assert(nvshmem_buffer != nullptr && "Failed to allocate NVSHMEM buffer");
    
    cudaMemset(nvshmem_buffer, 0, buffer_size);
    
    if (my_pe == 0) {
        std::cout << "\nCalibrating clocks..." << std::endl;
    }
    
    uint8_t* ping_buffer = (uint8_t*)nvshmem_malloc(1);
    uint64_t clock_offset = calibrate_clocks(my_pe, peer_pe, ping_buffer);
    nvshmem_free(ping_buffer);
    
    nvshmem_barrier_all();
    
    // Allocate metrics array in symmetric memory so both PEs can access it
    size_t metrics_size = 8 * n_runs * num_pairs * sizeof(uint64_t);
    uint64_t* d_metrics = (uint64_t*)nvshmem_malloc(metrics_size);
    assert(d_metrics != nullptr && "Failed to allocate NVSHMEM metrics buffer");
    cudaMemset(d_metrics, 0, metrics_size);
    
    if (my_pe == 0) {
        std::cout << "\nLaunching cross-node echo kernel..." << std::endl;
    }
    
    dim3 blockDim(32);
    dim3 gridDim(1);
    cross_node_echo_kernel<<<gridDim, blockDim>>>(
        d_metrics,
        nvshmem_buffer,
        msg_size,
        num_pairs,
        n_runs,
        my_pe,
        peer_pe,
        clock_offset
    );
    cudaDeviceSynchronize();
    
    nvshmem_barrier_all();
    
    if (my_pe == 0) {
        std::cout << "Cross-node echo kernel completed." << std::endl;
    }
    
    if (my_pe == 0) {
        std::cout << "\nWriting metrics to JSON file..." << std::endl;
        
        std::vector<uint64_t> h_metrics(8 * n_runs * num_pairs);
        cudaMemcpy(h_metrics.data(), d_metrics, metrics_size, cudaMemcpyDeviceToHost);
        
        nlohmann::json json_output;
        json_output["message_size"] = msg_size;
        json_output["num_pairs"]    = num_pairs;
        json_output["n_runs"]       = n_runs;
        json_output["clock_offset"] = clock_offset;
        json_output["communication_type"] = "cross_node_nvshmem";
        
        for (int run = 0; run < n_runs; ++run) {
            nlohmann::json run_json;
            for (int pair = 0; pair < num_pairs; ++pair) {
                int base_idx = (run * num_pairs + pair) * 8;
                nlohmann::json pair_json;
                
                pair_json["client_trans_start"] = h_metrics[base_idx + 0];
                pair_json["client_trans_end"]   = h_metrics[base_idx + 1];
                pair_json["client_recv_start"]  = h_metrics[base_idx + 2];
                pair_json["client_recv_end"]    = h_metrics[base_idx + 3];
                
                pair_json["server_recv_start"]  = h_metrics[base_idx + 4];
                pair_json["server_recv_end"]    = h_metrics[base_idx + 5];
                pair_json["server_trans_start"] = h_metrics[base_idx + 6];
                pair_json["server_trans_end"]   = h_metrics[base_idx + 7];
                
                run_json["pair" + std::to_string(pair)] = pair_json;
            }
            json_output["run" + std::to_string(run)] = run_json;
        }
        
        std::string deviceName = std::string(deviceProp.name);
        std::replace(deviceName.begin(), deviceName.end(), ' ', '_');
        create_directories("data/" + deviceName);
        std::string filename = "data/" + deviceName + "/metrics_" + 
                              std::to_string(msg_size) + "B_" + 
                              std::to_string(num_pairs) + "P.json";
        
        std::ofstream file(filename);
        file << json_output.dump(4);
        file.close();
        
        std::cout << "Metrics written to " << filename << std::endl;
    }
    
    nvshmem_free(d_metrics);
    nvshmem_free(nvshmem_buffer);
    nvshmem_finalize();
    
    return 0;
}
