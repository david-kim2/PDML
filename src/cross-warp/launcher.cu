#include <nlohmann/json.hpp>
#include <cuda_runtime.h>
#include <common.h>
#include <iostream>
#include <cassert>
#include <fstream>
#include <vector>


__global__ void cross_warp_echo_kernel(uint32_t* metrics, int msg_size, int num_pairs, int n_runs);
__global__ void warmup_kernel(float* A, float* B, float* C, int N);


int main() {
    // WARMUP KERNEL LAUNCH
    int N = 1024; // Matrix Dimension
    size_t bytes = N * N * sizeof(float);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    warmup_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // CROSS-WARP ECHO KERNEL LAUNCH
    int msg_size  = 1024; // Message size in bytes
    int num_pairs = 1;    // Number of warp pairs
    int n_runs    = 100;  // Number of runs

    assert(msg_size > 0 && "Message size must be greater than 0");
    assert((msg_size & (msg_size - 1)) == 0 && "Message size must be a power of 2");
    assert(num_pairs > 0 && "Number of pairs must be greater than 0");
    assert(num_pairs <= 32 && "Number of pairs must be less than or equal to 32");
    assert(msg_size % num_pairs == 0 && "Message size must be divisible by number of pairs");
    assert(n_runs > 0 && "Number of runs must be greater than 0");

    int msg_size_thread    = msg_size / num_pairs;
    size_t shared_mem_size = (2 * num_pairs * msg_size_thread) + (2 * num_pairs);

    size_t metrics_size = 6 * n_runs * num_pairs * sizeof(uint32_t);
    uint32_t* d_metrics;
    cudaMalloc(&d_metrics, metrics_size);
    cudaMemset(d_metrics, 0, metrics_size);

    dim3 blockDim(64);
    dim3 gridDim(1);
    cross_warp_echo_kernel<<<gridDim, blockDim, shared_mem_size>>>(d_metrics, msg_size, num_pairs, n_runs);
    cudaDeviceSynchronize();

    // WRITE METRICS TO JSON FILE
    std::vector<uint32_t> h_metrics(6 * n_runs * num_pairs);
    cudaMemcpy(h_metrics.data(), d_metrics, metrics_size, cudaMemcpyDeviceToHost);

    nlohmann::json json_output;
    json_output["message_size"] = msg_size;
    json_output["num_pairs"] = num_pairs;
    json_output["n_runs"] = n_runs;

    for (int run = 0; run < n_runs; ++run) {
        nlohmann::json run_json;
        for (int pair = 0; pair < num_pairs; ++pair) {
            int base_idx = (run * num_pairs + pair) * 6;
            std::vector<uint32_t> metrics_for_pair(
                h_metrics.begin() + base_idx,
                h_metrics.begin() + base_idx + 6
            );
            run_json["pair_" + std::to_string(pair)] = metrics_for_pair;
        }
        json_output["run_" + std::to_string(run)] = run_json;
    }

    std::ofstream file("metrics_output.json");
    file << json_output.dump(4);
    file.close();
    return 0;
}
