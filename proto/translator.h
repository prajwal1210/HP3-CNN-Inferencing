/************************************************************
 * translator.h:									                          *
 * Translates the protobuf objects from custom specification*
 * to operation objects defined in forward/operations.h     *
 *                                                          *
 * Author: Tanay Bhartia, Prajwal Singhania                 *
 ************************************************************/

#ifndef TRANSLATOR_H
#define TRANSLATOR_H

#include <stdio.h>
#include <iostream>
#include <vector>

#include <cuda.h>
#include <cudnn.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>

#include "forward/operations.h"
#include "proto/network.pb.h"

using namespace std;
/*
 * Class for Translating the Proto objects to operation objects defined in opearations.h
 */
class Translator
{
 public:
   /* Conv2D Layer Translator:
   *  Translates the DeepNet::Layer layer object of type Conv2D to the the operation Conv2D object 
   *  Input - Reference to the layer object, CUDNN Handle, Input Height that will be used for the forward pass being run, Input width,
   *          Batchsize, Convolution Algorithm to use 
   */
  Conv2D* translateConv2D_layer(DeepNet::Layer& layer, cudnnHandle_t cudnn, int input_height, int input_width, int batchsize, customAlgorithmType algo);
  
  /* Pool2D Layer Translator:
   *  Translates the DeepNet::Layer layer object of type Pool2D to the the operation Pool object 
   *  Input - Reference to the layer object, CUDNN Handle, Batchsize, Input channels, Input Height, Input width      
   */
  Pool* translatePool_layer(DeepNet::Layer& layer, cudnnHandle_t cudnn, int batchsize, int in_channels, int input_height, int input_weight);
  
  /* AdaptivePool2D Layer Translator:
   *  Translates the DeepNet::Layer layer object of type AdaptivePool2D to the the operation Pool object 
   *  Input - Reference to the layer object, CUDNN Handle, Batchsize, Input channels, Input Height, Input width      
   */
  Pool* translateAdaptivePool_layer(DeepNet::Layer& layer, cudnnHandle_t cudnn, int batchsize, int in_channels, int input_height, int input_weight);
  
  /* Activation Layer Translator:
   *  Translates the DeepNet::Layer layer object of type Activavtion to the the operation Activation object 
   *  Input - Reference to the layer object, CUDNN Handle, Batchsize, Input channels, Input Height, Input width      
   */
  Activation* translateActivation_layer(DeepNet::Layer& layer, cudnnHandle_t cudnn, int batchsize, int in_channels, int input_height, int input_width);
  
  /* Linear Layer Translator:
   *  Translates the DeepNet::Layer layer object of type Linear to the the operation Linear object 
   *  Input - Reference to the layer object, CUDNN Handle, CUBLAS Handle, batchsize
   */
  Linear* translateLinear_layer(DeepNet::Layer& layer, cudnnHandle_t cudnn, cublasHandle_t cublas, int batchsize);
};

#endif