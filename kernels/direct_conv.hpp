#ifndef __DIRECTCONV_H_		/*include guard*/
#define __DIRECTCONV_H_

#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>

/*declaring forward pass function and class*/

class passforward{
public:
	int out_channels;
	int input_channels;
	int kernel_height;
	int kernel_width;
	int padding;
	int stride;
	int batchsize_of_data;
	int input_height;
	int input_width;

	float* kernel_weights;
	float* input;

	passforward(int out_channels, int input_channels, int kernel_height, int kernel_width, int padding, int stride, float* kernel_weights,int batchsize_of_data, int input_height, int input_width, float* input);

	~passforward();
};

#endif

