#include <nlohmann/json.hpp>
#include <cuda_runtime.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <iostream>
#include <cassert>
#include <fstream>
#include <vector>
#include <algorithm>
#include <filesystem>

// Forward declarations
__global__ void cross_node_echo_kernel(
    uint64_t *metrics,
    uint8_t *nvshmem_buffer,
    int msg_size,
    int num_pairs,
    int n_runs,
    int my_pe,
    int peer_pe,
    int64_t clock_offset
);

__global__ void ping_pong_kernel(
    uint64_t* timestamps, 
    uint8_t* ping_buffer, 
    int my_pe, 
    int peer_pe
);

__global__ void warmup_kernel(float *A, float *B, float *C, int N);

__global__ void publish_offset_kernel(int64_t* offset_ptr, int64_t value, int peer_pe, int my_pe);

// Warmup kernel definition
#define TILE_SIZE 16

__global__ void warmup_kernel(float *A, float *B, float *C, int N)
{
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;
    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; t++)
    {
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

__global__ void publish_offset_kernel(int64_t* offset_ptr, int64_t value, int peer_pe, int my_pe)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    // write the offset into both PE locations (our own and the peer's)
    nvshmem_int64_p(offset_ptr, value, my_pe);
    nvshmem_int64_p(offset_ptr, value, peer_pe);
    nvshmem_quiet();
}

int64_t calibrate_clocks(int my_pe, int peer_pe, uint8_t *ping_buffer)
{
    const int num_calibrations = 20;

    // allocate a symmetric timestamps buffer (4 uint64 per trial will be written into PE0's buffer)
    uint64_t *d_timestamps = (uint64_t*)nvshmem_malloc(4 * sizeof(uint64_t));
    assert(d_timestamps != nullptr && "Failed to allocate symmetric timestamps buffer");
    cudaMemset(d_timestamps, 0, 4 * sizeof(uint64_t));

    // allocate a symmetric location to publish the final clock offset (as int64_t)
    int64_t *d_clock_offset = (int64_t*)nvshmem_malloc(sizeof(int64_t));
    assert(d_clock_offset != nullptr && "Failed to allocate symmetric clock offset buffer");
    int64_t zero = 0;
    cudaMemcpy(d_clock_offset, &zero, sizeof(int64_t), cudaMemcpyHostToDevice);

    std::vector<int64_t> offsets;

    for (int i = 0; i < num_calibrations; ++i)
    {
        // clear timestamps on PE0 before trial
        if (my_pe == 0)
            cudaMemset(d_timestamps, 0, 4 * sizeof(uint64_t));

        // run ping-pong; kernel will publish t0,t1,t2,t3 into PE0's d_timestamps indices
        ping_pong_kernel<<<1, 32>>>(d_timestamps, ping_buffer, my_pe, peer_pe);
        cudaDeviceSynchronize();

        if (my_pe == 0)
        {
            uint64_t h_timestamps[4];
            cudaMemcpy(h_timestamps, d_timestamps, 4 * sizeof(uint64_t), cudaMemcpyDeviceToHost);

            uint64_t t0 = h_timestamps[0];
            uint64_t t1 = h_timestamps[1];
            uint64_t t2 = h_timestamps[2];
            uint64_t t3 = h_timestamps[3];

            // if any timestamp is zero, skip this trial
            if (t0 == 0 || t1 == 0 || t2 == 0 || t3 == 0)
                continue;

            // offset = ((t1 - t0) - (t3 - t2)) / 2
            int64_t delta1 = (int64_t)(t1 - t0);
            int64_t delta2 = (int64_t)(t3 - t2);
            int64_t offset = (delta1 - delta2) / 2;
            offsets.push_back(offset);
        }

        nvshmem_barrier_all(); // sync before next trial
    }

    int64_t clock_offset = 0;
    if (my_pe == 0)
    {
        if (!offsets.empty())
        {
            std::sort(offsets.begin(), offsets.end());
            int mid = offsets.size() / 2;
            int64_t median_offset = offsets[mid];
            clock_offset = median_offset;  // keep as signed

            std::cout << "Median clock offset (PE1 - PE0) : " << median_offset << " ticks" << std::endl;

            // publish median_offset into peer's symmetric storage and into our own so both PEs can read it
            publish_offset_kernel<<<1,1>>>(d_clock_offset, clock_offset, peer_pe, my_pe);
            cudaDeviceSynchronize();
            nvshmem_barrier_all();
        }
        else
        {
            std::cout << "Warning: insufficient successful calibration trials; defaulting clock offset to 0" << std::endl;
            // still barrier so peer doesn't hang
            publish_offset_kernel<<<1,1>>>(d_clock_offset, (int64_t)0, peer_pe, my_pe);
            cudaDeviceSynchronize();
            nvshmem_barrier_all();
        }

        // copy the final published offset back to host
        int64_t host_offset = 0;
        cudaMemcpy(&host_offset, d_clock_offset, sizeof(int64_t), cudaMemcpyDeviceToHost);
        clock_offset = host_offset;
    }
    else
    {
        // other PEs wait on barrier; then read the published offset from their local symmetric pointer
        nvshmem_barrier_all();
        int64_t host_offset = 0;
        cudaMemcpy(&host_offset, d_clock_offset, sizeof(int64_t), cudaMemcpyDeviceToHost);
        clock_offset = host_offset;
    }

    // clean up symmetric allocs
    nvshmem_free(d_timestamps);
    nvshmem_free(d_clock_offset);

    return clock_offset;
}


int main(int argc, char **argv)
{
    // Configure NVSHMEM for cross-node communication
    setenv("NVSHMEM_BOOTSTRAP_TWO_STAGE", "1", 1);
    setenv("NVSHMEM_BOOTSTRAP_PMI", "PMI-2", 1);
    setenv("NVSHMEM_DISABLE_CUDA_VMM", "1", 1);  // Critical for cross-node

    // Set CUDA device based on local rank (before nvshmem_init)
    int local_rank = 0;
    char *local_rank_str = getenv("SLURM_LOCALID");
    if (local_rank_str)
    {
        local_rank = atoi(local_rank_str);
    }
    cudaSetDevice(local_rank);

    // Initialize NVSHMEM
    nvshmem_init();

    int my_pe = nvshmem_my_pe();
    int num_pes = nvshmem_n_pes();

    printf("PE %d of %d initialized\n", my_pe, num_pes);
    fflush(stdout);

    if (num_pes != 2)
    {
        if (my_pe == 0)
            std::cerr << "Error: Need 2 PEs for cross-node communication (got " << num_pes << ")" << std::endl;
        nvshmem_finalize();
        return -1;
    }

    // GET DEVICE PROPERTIES
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, local_rank);

    if (my_pe == 0)
    {
        std::cout << "============================================" << std::endl;
        std::cout << "Cross-Node Echo Benchmark" << std::endl;
        std::cout << "============================================" << std::endl;
        std::cout << "Running On Device: " << deviceProp.name << std::endl;
        std::cout << "Total Global Memory: " << deviceProp.totalGlobalMem / (1024 * 1024) << " MiB" << std::endl;
        std::cout << "Number of PEs: " << num_pes << std::endl;
        std::cout << std::endl;
    }

    // WARMUP KERNEL
    if (my_pe == 0)
        std::cout << "Launching warmup kernel..." << std::endl;

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

    if (my_pe == 0)
    {
        std::cout << "Warmup kernel completed." << std::endl;
        std::cout << std::endl;
        std::cout << "Launching cross-node echo kernel..." << std::endl;
    }   

    // Parse command line arguments
    int msg_size = 1024; // message size (bytes)
    int num_pairs = 1;   // number of thread pairs
    int n_runs = 10;     // number of runs

    if (argc == 4)
    {
        msg_size = std::stoi(argv[1]);
        num_pairs = std::stoi(argv[2]);
        n_runs = std::stoi(argv[3]);
    }
    else if (my_pe == 0)
    {
        std::cout << "Using default parameters: " << std::endl;
        std::cout << "Message Size: " << msg_size << " bytes" << std::endl;
        std::cout << "Number of Pairs: " << num_pairs << std::endl;
        std::cout << "Number of Runs: " << n_runs << std::endl;
    }

    // Validate parameters
    assert(msg_size > 0 && "Message size must be greater than 0");
    assert((msg_size & (msg_size - 1)) == 0 && "Message size must be a power of 2");
    assert(num_pairs > 0 && "Number of pairs must be greater than 0");
    assert(num_pairs <= 1024 && "Number of pairs must be less than or equal to 32");
    assert(msg_size % num_pairs == 0 && "Message size must be divisible by number of pairs");
    assert(n_runs > 0 && "Number of runs must be greater than 0");

    int msg_size_thread = msg_size / num_pairs;

    // Allocate NVSHMEM symmetric buffer (messages + flags)
    size_t buffer_size = (num_pairs * msg_size_thread) + num_pairs;
    uint8_t* d_nvshmem_buffer = (uint8_t*)nvshmem_malloc(buffer_size);
    assert(d_nvshmem_buffer != nullptr && "Failed to allocate NVSHMEM buffer");
    cudaMemset(d_nvshmem_buffer, 0, buffer_size);

    // Allocate metrics array (symmetric)
    size_t metrics_size = 8 * n_runs * num_pairs * sizeof(uint64_t);
    uint64_t *d_metrics = (uint64_t *)nvshmem_malloc(metrics_size);
    assert(d_metrics != nullptr && "Failed to allocate NVSHMEM metrics buffer");
    cudaMemset(d_metrics, 0, metrics_size);

    if (my_pe == 0)
        std::cout << "\nCalibrating clocks..." << std::endl;

    // Clock calibration
    uint8_t* ping_buffer = (uint8_t*)nvshmem_malloc(1);
    cudaMemset(ping_buffer, 0, 1);
    int peer_pe = (my_pe == 0) ? 1 : 0;
    int64_t clock_offset = calibrate_clocks(my_pe, peer_pe, ping_buffer);
    nvshmem_free(ping_buffer);

    // Launch kernel on both PEs
    dim3 blockDim(num_pairs);
    dim3 gridDim(1);

    nvshmem_barrier_all(); // Ensure all PEs are ready

    if (my_pe == 0)
        std::cout << "Launching cross-node echo kernel with message size " << msg_size 
                  << " bytes, " << num_pairs << " pairs, " << n_runs << " runs..." << std::endl;

    cross_node_echo_kernel<<<gridDim, blockDim>>>(
        d_metrics, 
        d_nvshmem_buffer,
        msg_size,
        num_pairs, 
        n_runs,
        my_pe, 
        peer_pe, 
        clock_offset
    );
    cudaDeviceSynchronize();

    nvshmem_barrier_all(); // Ensure all PEs complete

    if (my_pe == 0)
    {
        std::cout << "Cross-node echo kernel completed." << std::endl;
        std::cout << std::endl;
    }

    // Write metrics to JSON (only PE 0)
    if (my_pe == 0)
    {
        std::cout << "Writing metrics to JSON file..." << std::endl;

        std::vector<uint64_t> h_metrics(8 * n_runs * num_pairs);
        cudaMemcpy(h_metrics.data(), d_metrics, metrics_size, cudaMemcpyDeviceToHost);

        nlohmann::json json_output;
        json_output["message_size"] = msg_size;
        json_output["num_pairs"] = num_pairs;
        json_output["n_runs"] = n_runs;
        json_output["clock_offset_ticks"] = (int64_t)clock_offset;
        json_output["communication_type"] = "cross_node_nvshmem";

        for (int run = 0; run < n_runs; ++run)
        {
            nlohmann::json run_json;
            for (int pair = 0; pair < num_pairs; ++pair)
            {
                int base_idx = (run * num_pairs + pair) * 8;
                nlohmann::json pair_json;

                // client metrics from PE 0
                pair_json["client_trans_start"] = h_metrics[base_idx + 0];
                pair_json["client_trans_end"] = h_metrics[base_idx + 1];
                pair_json["client_recv_start"] = h_metrics[base_idx + 2];
                pair_json["client_recv_end"] = h_metrics[base_idx + 3];

                // server metrics
                pair_json["server_recv_start"] = h_metrics[base_idx + 4];
                pair_json["server_recv_end"] = h_metrics[base_idx + 5];
                pair_json["server_trans_start"] = h_metrics[base_idx + 6];
                pair_json["server_trans_end"] = h_metrics[base_idx + 7];

                run_json["pair" + std::to_string(pair)] = pair_json;
            }
            json_output["run" + std::to_string(run)] = run_json;
        }

        std::string deviceName = std::string(deviceProp.name);
        std::replace(deviceName.begin(), deviceName.end(), ' ', '_');
        std::filesystem::create_directories("data/" + deviceName);
        std::string name = "data/" + deviceName + "/crossnode_metrics_" + 
                          std::to_string(msg_size) + "B_" + std::to_string(num_pairs) + "P.json";

        std::ofstream file(name);
        file << json_output.dump(4);
        file.close();

        std::cout << "Metrics written to " << name << std::endl;
    }

    // Cleanup
    nvshmem_free(d_metrics);
    nvshmem_free(d_nvshmem_buffer);
    nvshmem_finalize();
    
    return 0;
}
