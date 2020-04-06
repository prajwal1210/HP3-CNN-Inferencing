#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>

float* forward(int out_size, int channel, int kernel_height, int kernel_width, int pad, int stride, float* kernel,
 int batch_size, int height, int width, float* input_layer);
