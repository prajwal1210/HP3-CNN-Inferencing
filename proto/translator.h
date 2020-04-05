#ifndef TRANSLATOR_H
#define TRANSLATOR_H

#include <cuda.h>
#include <cudnn.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>
#include "forward/operations.h"
#include "proto/network.pb.h"

using namespace std;

class Translator
{
 public:
  Conv2D* translateConv2D_layer(DeepNet::Layer& layer, cudnnHandle_t cudnn, int input_height, int input_width, int batchsize);
  Pool* translatePool_layer(DeepNet::Layer& layer, cudnnHandle_t cudnn, int batchsize, int in_channels, int input_height, int input_weight);
  Activation* translateActivation_layer(DeepNet::Layer& layer, cudnnHandle_t cudnn, int batchsize, int in_channels, int input_height, int input_width);
  Linear* translateLinear_layer(DeepNet::Layer& layer, cudnnHandle_t cudnn, cublasHandle_t cublas, int batchsize);
};

#endif