#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>
#include <algorithm>
#include <random>

void gpu_error(cudaError_t const&);

template<typename T> __device__ inline void __swap(T&, T&);

__global__ void solve(float *, size_t);
