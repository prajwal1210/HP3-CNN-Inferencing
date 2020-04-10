
#ifndef WINGHEADER_H 
#define WINGHEADER_H

#include <iostream>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace WING {
  /* Forward pass of Winograd Kernel:
   * Computes the convolution on the given input and kernel using Winograd method
   * Input - Output channels(och), Input Channels(ch), Batch Size(bs), Input Height(h), Input Width(w), Padding(pad), 
   			 Pointer to input unpadded array(in), Output Height(oph), Output Width(opw), Kernel Weights(kwt)
   *         Pointer to input array
   * Output - The output of the convolution       
   */
	float *forward(int och, int ch, int bs, int h, int w, int pad, float *&in, int &oph, int &opw, float *kwt);
}

#endif