#ifndef DIRECT_CONV_H
#define DIRECT_CONV_H

#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>


namespace Direct {
	/* Forward pass of Direct Convolution Kernel:
   * Computes the convolution on the given input and kernel Direct Convolution method (input and output are device arrays and not host)
   * Input - Output channels, Input Channels, Kernel Height, Kernel Width, Padding, Stride, Kernel Weights (ON DEVICE), Batch Size, Input Height, Input Width,
   *         Input Array (ON DEVICE)
   * Output - The output of the convolution       
   */
	float* passforward(int out_channels, int input_channels, int kernel_height, int kernel_width, int padding, int stride, 
					          float* d_weights,int batchsize_of_data, int input_height, int input_width, float* d_input);
}

#endif

