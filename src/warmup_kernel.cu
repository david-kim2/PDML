#include "common.hpp"
#include <stdint.h>
#include <cstdio>
#include <cuda.h>


__global__ void warmup_kernel(float* A, float* B, float* C, int N) {
    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;

    for (int tile = 0; tile < (N + TILE_SIZE - 1) / TILE_SIZE; tile++) {
        // Load tiles into shared memory (with boundary checks)
        if (row < N && tile * TILE_SIZE + threadIdx.x < N)
            sA[threadIdx.y][threadIdx.x] = A[row * N + tile * TILE_SIZE + threadIdx.x];
        else sA[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < N && tile * TILE_SIZE + threadIdx.y < N)
            sB[threadIdx.y][threadIdx.x] = B[(tile * TILE_SIZE + threadIdx.y) * N + col];
        else sB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // Multiply the two tiles
        for (int k = 0; k < TILE_SIZE; k++)
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        __syncthreads();
    }

    // Write the result
    if (row < N && col < N)
        C[row * N + col] = sum;
}
