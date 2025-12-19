#include <nlohmann/json.hpp>
#include <cuda_runtime.h>
#include "../common.hpp"
#include <filesystem>
#include <algorithm>
#include <iostream>
#include <cassert>
#include <fstream>
#include <vector>


__global__ void uint64_cross_block_echo_kernel(uint64_t* metrics, uint64_t* client_buf, uint64_t* server_buf,
                                        volatile uint8_t* finished_c2s, volatile uint8_t* finished_s2c,
                                        size_t msg_size, int num_pairs, int n_runs);
__global__ void warmup_kernel(float* A, float* B, float* C, int N);


int main(int argc, char** argv) {
    // GET DEVICE PROPERTIES
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    int maxThreads = deviceProp.maxThreadsPerBlock;

    std::cout << "============================================" << std::endl;
    std::cout << "Running On Device: " << deviceProp.name << std::endl;
    std::cout << "Total Global Memory: " << deviceProp.totalGlobalMem / (1024 * 1024) << " MiB" << std::endl;
    std::cout << "Max Threads Per Block: " << maxThreads << std::endl;
    std::cout << std::endl;

    int supportsCoopLaunch = 0;
    cudaDeviceGetAttribute(&supportsCoopLaunch, cudaDevAttrCooperativeLaunch, 0);
    if (!supportsCoopLaunch) {
        std::cerr << "Device does NOT support cooperative launch." << std::endl;
        return -1;
    }

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

    // CROSS-BLOCK ECHO KERNEL LAUNCH
    std::cout << "Launching cross-block echo kernel..." << std::endl;
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
    assert(num_pairs > 0 && "Number of pairs must be greater than 0");
    assert(num_pairs <= maxThreads && "Number of pairs must be less than or equal to maxThreads");
    assert(msg_size % num_pairs == 0 && "Message size must be divisible by number of pairs");
    assert(n_runs > 0 && "Number of runs must be greater than 0");

    size_t msg_size_thread = msg_size / num_pairs;
    size_t msg_size_word   = (msg_size_thread + 7) / 8;
    size_t buf_bytes       = num_pairs * msg_size_word * sizeof(uint64_t);
    size_t global_mem_size = 2 * (buf_bytes) + 2 * num_pairs; // client_buf + server_buf + finished_c2s + finisheds2c
    assert(global_mem_size <= deviceProp.totalGlobalMem && "Global memory size exceeds device limit");

    uint16_t* d_client_buf;
    uint16_t* d_server_buf;
    cudaMalloc(&d_client_buf, buf_bytes);
    cudaMalloc(&d_server_buf, buf_bytes);
    cudaMemset(d_client_buf, 0, buf_bytes);
    cudaMemset(d_server_buf, 0, buf_bytes);

    uint8_t* d_finished_c2s;
    uint8_t* d_finished_s2c;
    cudaMalloc(&d_finished_c2s, num_pairs);
    cudaMalloc(&d_finished_s2c, num_pairs);
    cudaMemset(d_finished_c2s, 0, num_pairs);
    cudaMemset(d_finished_s2c, 0, num_pairs);

    size_t metrics_size = 8 * n_runs * num_pairs * sizeof(uint64_t);
    uint64_t* d_metrics;
    cudaMalloc(&d_metrics, metrics_size);
    cudaMemset(d_metrics, 0, metrics_size);

    int threads = (num_pairs + 31) / 32 * 32; // Round up to the nearest warp
    dim3 blockDim(threads);
    dim3 gridDim(2);
    void* kernelArgs[] = {
        (void*)&d_metrics, (void*)&d_client_buf, (void*)&d_server_buf,
        (void*)&d_finished_c2s, (void*)&d_finished_s2c,
        (void*)&msg_size, (void*)&num_pairs, (void*)&n_runs
    };

    cudaError_t err = cudaLaunchCooperativeKernel((void*)uint64_cross_block_echo_kernel, gridDim, blockDim, kernelArgs);
    if (err != cudaSuccess) {
        std::cerr << "Cooperative kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    cudaDeviceSynchronize();
    cudaFree(d_client_buf);
    cudaFree(d_server_buf);
    cudaFree(d_finished_c2s);
    cudaFree(d_finished_s2c);
    std::cout << "Cross-block echo kernel completed." << std::endl;
    std::cout << std::endl;

    // WRITE METRICS TO JSON FILE
    std::cout << "Writing metrics to JSON file..." << std::endl;
    std::vector<uint64_t> h_metrics(8 * n_runs * num_pairs);
    cudaMemcpy(h_metrics.data(), d_metrics, metrics_size, cudaMemcpyDeviceToHost);

    nlohmann::json json_output;
    json_output["message_size"] = msg_size;
    json_output["num_pairs"]    = num_pairs;
    json_output["n_runs"]       = n_runs;

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
    std::replace(deviceName.begin(), deviceName.end(), ' ', '_'); // Replace spaces with underscores
    std::filesystem::create_directories("data/" + deviceName + "-uint64");
    std::string name = "data/" + deviceName + "-uint64/metrics_" + std::to_string(msg_size) + "B_" + std::to_string(num_pairs) + "P.json";

    std::ofstream file(name);
    file << json_output.dump(4);
    file.close();

    cudaFree(d_metrics);
    std::cout << "Metrics written to " << name << std::endl;
    return 0;
}
