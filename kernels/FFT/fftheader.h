#ifndef FFTHEADER_H 
#define FFTHEADER_H

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>

namespace FFT {
  /* Forward pass of FFT Kernel:
   * Computes the convolution on the given input and kernel using FFT method
   * Input - Output channels, Input Channels, Kernel Height, Kernel Width, Padding, Stride, Kernel Weights (on host), Batch Size, Input Height, Input Width,
   *         Pointer to input array
   * Output - The output of the convolution       
   */
  float* forward(int out_size, int channel, int kernel_height, int kernel_width, int pad, int stride, float* kernel,
              int batch_size, int height, int width, float* input_layer_without_padding, float& conv_time, float& overhead_time);
}

#endif