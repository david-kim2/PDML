#include <cuda_runtime.h>
#include "../common.hpp"
#include "../json.hpp"
#include <filesystem>
#include <iostream>
#include <cassert>
#include <fstream>
#include <vector>


__global__ void cross_warp_echo_kernel(uint64_t* metrics, int msg_size, int num_pairs, int n_runs);
__global__ void warmup_kernel(float* A, float* B, float* C, int N);


int main(int argc, char** argv) {
    // GET DEVICE PROPERTIES
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    std::cout << "============================================" << std::endl;
    std::cout << "Running On Device: " << deviceProp.name << std::endl;
    std::cout << "Max Shared Memory Per Block: " << deviceProp.sharedMemPerBlock / 1024 << " KiB" << std::endl;
    std::cout << std::endl;

    // WARMUP KERNEL LAUNCH
    std::cout << "Launching warmup kernel..." << std::endl;
    int N = 1024; // Matrix Dimension
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

    // CROSS-WARP ECHO KERNEL LAUNCH
    std::cout << "Launching cross-warp echo kernel..." << std::endl;
    int msg_size = 1024; // Default Message Size in bytes
    int num_pairs = 1;  // Default Number of warp pairs
    int n_runs = 10;    // Default Number of runs
    if (argc == 4) {
        msg_size  = std::stoi(argv[1]); // Message size in bytes
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
    assert(num_pairs <= 32 && "Number of pairs must be less than or equal to 32");
    assert(msg_size % num_pairs == 0 && "Message size must be divisible by number of pairs");
    assert(n_runs > 0 && "Number of runs must be greater than 0");

    int msg_size_thread    = msg_size / num_pairs;
    size_t shared_mem_size = (2 * num_pairs * msg_size_thread) + (2 * num_pairs);
    assert(shared_mem_size <= deviceProp.sharedMemPerBlock && "Shared memory size exceeds device limit");

    size_t metrics_size = 8 * n_runs * num_pairs * sizeof(uint64_t);
    uint64_t* d_metrics;
    cudaMalloc(&d_metrics, metrics_size);
    cudaMemset(d_metrics, 0, metrics_size);

    dim3 blockDim(64);
    dim3 gridDim(1);
    cross_warp_echo_kernel<<<gridDim, blockDim, shared_mem_size>>>(d_metrics, msg_size, num_pairs, n_runs);
    cudaDeviceSynchronize();
    std::cout << "Cross-warp echo kernel completed." << std::endl;
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

            pair_json["client_start"]       = h_metrics[base_idx + 0];
            pair_json["client_end"]         = h_metrics[base_idx + 1];
            pair_json["client_recv_start"]  = h_metrics[base_idx + 2];
            pair_json["client_recv_end"]    = h_metrics[base_idx + 3];

            pair_json["server_recv_start"] = h_metrics[base_idx + 4];
            pair_json["server_recv_end"]   = h_metrics[base_idx + 5];
            pair_json["server_start"]      = h_metrics[base_idx + 6];
            pair_json["server_end"]        = h_metrics[base_idx + 7];
            run_json["pair" + std::to_string(pair)] = pair_json;
        }
        json_output["run" + std::to_string(run)] = run_json;
    }

    std::string deviceName = std::string(deviceProp.name);
    std::replace(deviceName.begin(), deviceName.end(), ' ', '_'); // Replace spaces with underscores
    std::filesystem::create_directories("data/" + deviceName);
    std::string name = "data/" + deviceName + "/metrics_" + std::to_string(msg_size) + "B_" + std::to_string(num_pairs) + "P.json";

    std::ofstream file(name);
    file << json_output.dump(4);
    file.close();

    std::cout << "Metrics written to " << name << std::endl;
    return 0;
}
