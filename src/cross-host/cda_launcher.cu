#include <nlohmann/json.hpp>
#include <cuda_runtime.h>
#include <sys/sysinfo.h>
#include "../common.hpp"
#include <filesystem>
#include <algorithm>
#include <iostream>
#include <cassert>
#include <fstream>
#include <vector>
#include <chrono>

uint64_t now_ns() {
    auto now = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
}
__global__ void warmup_kernel(float* A, float* B, float* C, int N);


int main(int argc, char** argv) {
    // GET DEVICE PROPERTIES
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    std::cout << "============================================" << std::endl;
    std::cout << "Running On Device: " << deviceProp.name << std::endl;
    std::cout << "Total Global Memory: " << deviceProp.totalGlobalMem / (1024 * 1024) << " MiB" << std::endl;

    struct sysinfo info;
    sysinfo(&info);
    std::cout << "Host RAM Total: " << info.totalram / (1024 * 1024) << " MiB" << std::endl;
    std::cout << "Host RAM Free:  " << info.freeram / (1024 * 1024) << " MiB" << std::endl;
    std::cout << std::endl;

    // WARMUP KERNEL LAUNCH
    std::cout << "Launching warmup kernel..." << std::endl;
    int N = 8192; // Matrix Dimension
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
    std::cout << "Warmup kernel completed." << std::endl;
    std::cout << std::endl;

    // CROSS-HOST ECHO KERNEL LAUNCH
    std::cout << "Launching cross-host echo kernel..." << std::endl;
    size_t msg_size = 1024; // Default Message Size in bytes
    int num_pairs   = 1;    // Default Number of warp pairs
    int n_runs      = 10;   // Default Number of runs

    if (argc == 4) {
        msg_size  = std::stoull(argv[1]); // Message size in bytes
        num_pairs = std::stoi(argv[2]); // Number of warp pairs
        n_runs    = std::stoi(argv[3]); // Number of runs
    } else {
        std::cout << "Using default parameters: " << std::endl;
        std::cout << "Message Size: " << msg_size << " bytes" << std::endl;
        std::cout << "Number of Pairs: " << num_pairs << std::endl;
        std::cout << "Number of Runs: " << n_runs << std::endl;
    }

    assert(msg_size > 0 && "Message size must be greater than 0");
    assert((msg_size & (msg_size - 1)) == 0 && "Message size must be a power of 2");
    assert(num_pairs == 1 && "Number of pairs must be equal to 1");
    assert(msg_size % num_pairs == 0 && "Message size must be divisible by number of pairs");
    assert(n_runs > 0 && "Number of runs must be greater than 0");

    assert(2 * msg_size <= info.freeram && "Message size exceeds available host RAM");
    assert(msg_size <= deviceProp.totalGlobalMem && "Message size exceeds device global memory");

    uint8_t* h_buffer_send = (uint8_t*) malloc(msg_size);
    memset(h_buffer_send, 0xFF, msg_size);
    uint8_t* h_buffer_recv = (uint8_t*) malloc(msg_size);
    memset(h_buffer_recv, 0x00, msg_size);
    
    uint8_t* d_buffer;
    cudaMalloc(&d_buffer, msg_size);
    cudaMemset(d_buffer, 0x00, msg_size);
    cudaDeviceSynchronize();

    nlohmann::json json_output;
    json_output["message_size"] = msg_size;
    json_output["num_pairs"]    = num_pairs;
    json_output["n_runs"]       = n_runs;

    for (int run = 0; run < n_runs; ++run) {
        uint64_t trans_start, trans_end, recv_start, recv_end;
        
        // Transfer
        trans_start = now_ns();
        cudaMemcpy(d_buffer, h_buffer_send, msg_size, cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
        trans_end   = now_ns();
        
        // Receive
        recv_start = now_ns();
        cudaMemcpy(h_buffer_recv, d_buffer, msg_size, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        recv_end   = now_ns();

        // Verify data
        bool valid = true;
        for (size_t i = 0; i < msg_size; i++) {
            valid &= (h_buffer_recv[i] == 0xFF);
        }

        // Record metrics
        nlohmann::json run_json;
        if (valid) {
            run_json["client_trans_start"] = trans_start;
            run_json["client_trans_end"]   = trans_end;
            run_json["client_recv_start"]  = recv_start;
            run_json["client_recv_end"]    = recv_end;
        } else {
            run_json["client_trans_start"] = 0ull;
            run_json["client_trans_end"]   = 0ull;
            run_json["client_recv_start"]  = 0ull;
            run_json["client_recv_end"]    = 0ull;
        }
        json_output["run" + std::to_string(run)] = run_json;

        // Reset buffers
        memset(h_buffer_recv, 0x00, msg_size);
        cudaMemset(d_buffer, 0x00, msg_size);
        cudaDeviceSynchronize();
    }

    cudaDeviceSynchronize();
    free(h_buffer_send);
    free(h_buffer_recv);
    cudaFree(d_buffer);
    std::cout << "Cross-host echo kernel completed." << std::endl;
    std::cout << std::endl;

    // WRITE METRICS TO JSON FILE
    std::cout << "Writing metrics to JSON file..." << std::endl;
    std::string deviceName = std::string(deviceProp.name);
    std::replace(deviceName.begin(), deviceName.end(), ' ', '_'); // Replace spaces with underscores
    std::filesystem::create_directories("data/" + deviceName);
    std::string name = "data/" + deviceName + "/cda_metrics_" + std::to_string(msg_size) + "B_" + std::to_string(num_pairs) + "P.json";

    std::ofstream file(name);
    file << json_output.dump(4);
    file.close();

    std::cout << "Metrics written to " << name << std::endl;
    return 0;
}
