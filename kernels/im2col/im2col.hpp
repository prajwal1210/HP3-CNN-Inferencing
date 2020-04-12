#ifndef __IM2COL_H__
#define __IM2COL_H__

#include "common.h"

int check_result(float* a, float* b, int size);

cudaError_t addWithCuda(float *c, float *a, float *b, unsigned int size);

cudaError_t im2colWithCuda(
	const float* data_im,
	const int batch_size,
	const int channels,
	const int height, const int width, const int ksize, const int pad,
	const int stride,
	float* data_col,
	const int num_kernels,
	float* data_kernel,
	float* data_ret);

cudaError_t bu_im2colWithCuda(
	const float* data_im,
	const int batch_size,
	const int channels,
	const int height, const int width, const int ksize, const int pad,
	const int stride,
	float* data_col,
	const int num_kernels,
	float* data_kernel,
	float* data_ret);

namespace IM2COL
{
	/* Forward pass of Im2Col Kernel:
	* Computes the convolution on the given input and kernel using Im2Col followed by GEMM
	* Input - Output channels, Input Channels, Kernel Height, Kernel Width, Padding, Stride, Kernel Weights (on host), Batch Size, Input Height, Input Width,
	*		 Pointer to input array
	* Output - The output of the convolution	   
	*/
	float* forward(int out_size, int channel, int kernel_height, int kernel_width, int pad, int stride, 
		float* kernel, int batch_size, int input_height, int input_width, float* input);
}

#endif // __IM2COL_H__

im2colWithCuda(input, batch_size, channel, input_height, input_height, 
	kernel_height, pad, stride, nullptr, out_size, kernel, nullptr);

