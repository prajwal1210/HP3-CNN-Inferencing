#ifndef __IM2COL_H__
#define __IM2COL_H__

#include <iostream>
#include <cstdlib>
#include <string>

// CUDA and CUBLAS runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>


/* Macro to check CUDA error and print the error message */
#define CUDA_CHECK(condition) {									\
	cudaError_t error = (condition);							\
	if (error != cudaSuccess) {									\
		std::cerr << "CUDA Error on line " << __LINE__ << ": "	\
				  << cudaGetErrorString(error) << std::endl;	\
		std::exit(EXIT_FAILURE);								\
	}															\
}

#define CUDA_POST_KERNEL_CHECK CUDA_CHECK(cudaPeekAtLastError())

/* Function to check cuBLAS error and print the error message */
void inline CUBLAS_CHECK(cublasStatus_t status, std::string msg)
{
	if (status != CUBLAS_STATUS_SUCCESS)
	{
		std::cerr << msg << " at line " << __LINE__ << std::endl;
		std::exit(EXIT_FAILURE);
	}
}

// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n) \
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
	i < (n); \
	i += blockDim.x * gridDim.x)


// CUDA: thread number configuration.
// Use 1024 threads per block, which requires cuda sm_2x or above,
// or fall back to attempt compatibility (best of luck to you).
#if __CUDA_ARCH__ >= 200
	const int CUDA_NUM_THREADS = 1024;
#else
	const int CUDA_NUM_THREADS = 512;
#endif

// CUDA: number of blocks for N threads, given CUDA_NUM_THREADS per block.
inline int GET_BLOCKS(const int N) {
	return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS; // basically ceil(N/CUDA_NUM_THREADS)
}

namespace IM2COL
{
	/* Forward pass of Im2Col Kernel:
	* Computes the convolution on the given input and kernel using Im2Col followed by GEMM
	* Input - Output channels, Input Channels, Kernel Height, Kernel Width, Padding, Stride, Kernel Weights (on host), Batch Size, Input Height, Input Width,
	*		 Pointer to input array
	* Output - The output of the convolution	   
	*/
	float* forward(int out_size, int channel, int kernel_height, int kernel_width, int pad, int stride, 
		float* kernel, int batch_size, int input_height, int input_width, float* input, float& conv_time, float& overhead_time);
}

#endif // __IM2COL_H__