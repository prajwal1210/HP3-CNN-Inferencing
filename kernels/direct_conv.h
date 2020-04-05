#ifndef __DIRECTCONV_H_
#define __DIRECTCONV_H_

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>

void passforward(int out_channels, int input_channels, int kernel_height, int kernel_width, int padding, int stride, float* kernel_weights
		int batchsize_of_data, int input_height, int input_width, float* input);

#endif