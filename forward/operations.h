/************************************************************
 * operations.h:									                          *
 * Library that contains forward pass implementation as     * 
 * classes for all the standard CNN operations like Conv,   *
 * Pool, Activation and FC		                              *
 *                                                          *
 * Author: Prajwal Singhania								                *
 ************************************************************/

#ifndef OPERATIONS_H 
#define OPERATIONS_H

#include <iostream>

#include <cuda.h>
#include <cudnn.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>
#include "kernels/Direct/direct_conv.h"


/* Macro to check CUDNN error and print the error message */
#define checkCUDNN(expression) {                             \
    cudnnStatus_t status = (expression);                     \
    if (status != CUDNN_STATUS_SUCCESS) {                    \
      std::cerr << "Error on line " << __LINE__ << ": "      \
                << cudnnGetErrorString(status) << std::endl; \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
}

/* Macro to check CUDA error and print the error message */
#define checkCudaErrors(expression) {                        \
    cublasStatus_t status = (expression);                    \
    if (status != 0) {                                       \
      std::cerr << "Error on line " << __LINE__ << ": "      \
                << status << std::endl;                      \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
}


/* Class for Convolution Operation :
 *  Supports Forward Pass with and without Bias
 */
class Conv2D {   
 public:
  int out_channels;
  int in_channels;
  int h;                  /* Height of the filters */
  int w;                  /* Width of the filters */
  int batchsize;
  int input_height;
  int input_width;

  int padding;            /* Assumed same in x and y direction */
  int stride;             /* Assumed same in x and y direction */
  int dilation;           /* Assumed same in x and y direction */
  float* weights;        
  bool bias_present;      
  float* bias;            

  cudnnTensorFormat_t data_format = CUDNN_TENSOR_NCHW;
  cudnnTensorFormat_t param_format = CUDNN_TENSOR_NCHW;

  cudnnHandle_t cudnn;                                    
  cudnnTensorDescriptor_t input_descriptor;
  cudnnTensorDescriptor_t output_descriptor;
  cudnnFilterDescriptor_t kernel_descriptor;
  cudnnTensorDescriptor_t convbias_descriptor;
  cudnnConvolutionDescriptor_t convolution_descriptor;
  cudnnConvolutionFwdAlgo_t convolution_algorithm;
  size_t workspace_bytes = 0;


  
  /* Constructor:
   *  To create the Conv2D Class object
   *  Input - Output Channels (# filters), Input Channels, Filter Height, Filter Width, BatchSize of data, Padding, Stride, Dilation,
   *          Height of the Input, Width of the Input, CUDNN Descriptor
   */
  Conv2D(int out_channels, int in_channels, int h, int w, int batchsize, int padding, int stride, int dilation, 
        int input_height, int input_width, cudnnHandle_t cudnn);
  
  /* Destructor */
  ~Conv2D();

  /* Weight Initializer: 
   *  Used to set the value of the filter weights of the Conv2D layer
   *  Input - Pointer to the weight parameters
   */
  void SetWeights(float* weights);

  /* Bias Initializer: 
   *  Used to set the indicator (bool to indicate the presence of bias) and value of the bias of the Conv2D layer
   *  Input - Pointer to the bias parameters
   */
  void SetBias(float* bias);

  /* Get Output Dimensions:
   *  Returns the dimensions (in the passed references) of the output after convolution
   *  Input - Pointer to the variable storing output batchsize (n), Pointer to the variable storing output channels (c),
   *          Pointer to the variable storing output height (h), Pointer to the variable storing output width (w) 
   */
  void GetOutputDims(int* out_n, int* out_c, int* out_h, int* out_w);

  /* Forward Pass Operation:
   *  Computes the forward convolution on the given input
   *  Input - Values/Inputs from the previous layer (in host memory)
   *  Output - Pointer to the result array (in host memory)
   */
  float* ConvForward(float* input);

  float* Conv_Direct(float* input);

 private:
  /* Create Descriptors: 
   *  Creates the necessary descriptors based on the data members (helper to the constructor)
   */
  void CreateDescriptors();
};


typedef enum{
  t_max = 0,
  t_avg
} poolType;

/* Class for Pooling Operation :
 *  Supports both Max and Average Pooling (Only Forward Pass)
 */
class Pool {
 public:
  poolType type;       /* 0 - Max, 1 - Average */
  int in_channels;
  int batchsize;      
  int kernel_size_x;
  int kernel_size_y;
  int padding;
  int stride_x;
  int stride_y;
  int input_height;
  int input_width;
  
  cudnnTensorFormat_t data_format = CUDNN_TENSOR_NCHW;

  cudnnHandle_t cudnn;
  cudnnTensorDescriptor_t input_descriptor;
  cudnnTensorDescriptor_t output_descriptor;
  cudnnPoolingDescriptor_t pooling_descriptor;
  cudnnPoolingMode_t mode;

  
  /* Constructors */

  /* Constructor for Normal Pooling:
   *  Assumes same kernel size and stride in X and Y direction
   *  Input - Type of Pooling (0 - MAX, 1 - AVG), BatchSize of Data, Number of Input Channels, Height of the Input, Width of the Input, 
   *          Kernel Size (Same in X and Y), Padding, Stride Size (Same in X and Y), CUDNN Descriptor
   */
  Pool(int type, int batchsize, int in_channels, int input_height, int input_width, int kernel_size, int padding, int stride, cudnnHandle_t cudnn);

  /* Constructor for Normal Pooling:
   *  Allows different kernel sizes and stride in X and Y direction
   *  Input - Type of Pooling (0 - MAX, 1 - AVG), BatchSize of Data, Number of Input Channels, Height of the Input, Width of the Input, 
   *          Kernel Size in the Y direction, Kernel Size in the X direction, Padding, Stride Size in the Y direction, 
   *          Stride Size in the X direction, CUDNN Descriptor
   */
  Pool(int type, int batchsize, int in_channels, int input_height, int input_width, int kernel_size_y, int kernel_size_x, int padding,
      int stride_y, int stride_x, cudnnHandle_t cudnn);
  
  /* Constructor for Adaptive Pooling:
   *  Computes the parameters of pooling based on a fixed output size. Assumes same kernel size and stride in X and Y direction
   *  Input - Type of Pooling (0 - MAX, 1 - AVG), BatchSize of Data, Number of Input Channels, Height of the Input, Width of the Input, 
   *          Height of the Output, Width of the Output, CUDNN Descriptor
   */
  Pool(int type, int batchsize, int in_channels, int input_height, int input_width, int output_height, int output_width, cudnnHandle_t cudnn);
  
  /* Destructor */
  ~Pool();

  /* Get Output Dimensions:
   *  Returns the dimensions (in the passed references) of the output after pooling 
   *  Input - Pointer to the variable storing output batchsize (n), Pointer to the variable storing output channels (c),
   *          Pointer to the variable storing output height (h), Pointer to the variable storing output width (w) 
   */
  void GetOutputDims(int* out_n, int* out_c, int* out_h, int* out_w);

  /* Forward Pass Operation:
   *  Computes the forward pass of pooling on the given input
   *  Input - Values/Inputs from the previous layer (in host memory)
   *  Output - Pointer to the result array (in host memory)
   */
  float* PoolForward(float* input);

 private:
  /* Helper Function to the Constructors:
   *  Initializes the data members of object (Helper to the constructors to do the common job)
   *  Input - Type of Pooling (0 - MAX, 1 - AVG), BatchSize of Data, Number of Input Channels, Height of the Input, Width of the Input, 
   *          Kernel Size in the Y direction, Kernel Size in the X direction, Padding, Stride Size in the Y direction, 
   *          Stride Size in the X direction, CUDNN Descriptor
   */
  void InitalizeAttributes(int type, int batchsize, int in_channels, int input_height, int input_width, int kernel_size_y, int kernel_size_x, 
                          int padding, int stride_y, int stride_x, cudnnHandle_t cudnn);

  /* Create Descriptors: 
   *  Creates the necessary descriptors based on the data members (helper to the constructor)
   */
  void CreateDescriptors();
};


typedef enum{
  t_relu = 0,
  t_sigmoid
} actType;

/* Class for Activation Operation: 
 *  Supports ReLU and Sigmoid Activations
 */
class Activation {            
 public:
  actType type;       /* 0 - ReLU, 1 - Sigmoid */
  int batchsize;
  int in_channels;
  int input_height;
  int input_width;

  cudnnTensorFormat_t data_format = CUDNN_TENSOR_NCHW;

  cudnnHandle_t cudnn;
  cudnnTensorDescriptor_t output_descriptor;
  cudnnActivationDescriptor_t activation_descriptor;
  cudnnActivationMode_t mode;

  /* Constructor:
   *  To create the Activation Class object
   *  Input - Type of Activation (0 - ReLU, 1 - Sigmoid), BatchSize of the data, Number of Input Channels,
   *          Height of the Input, Width of the Input, CUDNN Descriptor
   */
  Activation(int type, int batchsize, int in_channels, int input_height, int input_width, cudnnHandle_t cudnn);                                               //Normal Pooling - Kernel and Stride can be different in x and y
  
  /* Destructor */
  ~Activation();

  /* Get Output Dimensions:
   *  Returns the dimensions (in the passed references) of the output after activation (will be same as the input dimensions)
   *  Input - Pointer to the variable storing output batchsize (n), Pointer to the variable storing output channels (c),
   *          Pointer to the variable storing output height (h), Pointer to the variable storing output width (w) 
   */
  void GetOutputDims(int* out_n, int* out_c, int* out_h, int* out_w);

  /* Forward Pass Operation:
   *  Computes the forward pass of activation on the given input
   *  Input - Values/Inputs from the previous layer (in host memory)
   *  Output - Pointer to the result array (in host memory)
   */
  float* ActivationForward(float* input);

 private:
  /* Create Descriptors: 
   *  Creates the necessary descriptors based on the data members (helper to the constructor)
   */
  void CreateDescriptors();
};


/* Class for Linear Layer:
 *  Supports Forward Pass of Linear (FC) Layer with and without bias
 */
class Linear {            
 public:
  int batchsize;
  int out_nodes;
  int in_nodes;
  float* weight;          /* Size: Out_nodes X In_nodes */
  bool bias_present;
  float* bias;            /* Size: Out_nodes X 1 */

  cudnnTensorFormat_t data_format = CUDNN_TENSOR_NCHW;

  cudnnHandle_t cudnn;
  cublasHandle_t cublas;
  cudnnTensorDescriptor_t bias_input_descriptor;      /* Input Descriptor for Bias Addition */
  cudnnTensorDescriptor_t bias_output_descriptor;     /* Output Descriptor for Bias Addition */
  cudnnTensorDescriptor_t bias_descriptor;
  
  /* Constructor:
   *  To create the Linear Layer Class object
   *  Input - BatchSize of the data, Output Nodes in the Linear Layer,
   *          Input Nodes in the Linear Layer, CUDNN Descriptor
   */
  Linear(int batchsize, int out_nodes, int in_nodes, cublasHandle_t cublas);
  
  /* Destructor */
  ~Linear();

  /* Get Output Dimensions:
   *  Returns the dimensions (in the passed references) of the output after the Linear Layer (out_h and out_w = 1)
   *  Input - Pointer to the variable storing output batchsize (n), Pointer to the variable storing output channels (c),
   *          Pointer to the variable storing output height (h), Pointer to the variable storing output width (w) 
   */
  void GetOutputDims(int* out_n, int* out_c, int* out_h, int* out_w);

  /* Weight Initializer: 
   *  Used to set the value of the weights of Linear Layer
   *  Input - Pointer to the weight parameters
   */
  void SetWeights(float* weights);

  /* Bias Initializer: 
   *  Used to set the indicator (bool to indicate the presence of bias) and value of the bias of the Linear Layer and set descriptors
   *  Input - Pointer to the bias parameters, CUDNN Descriptor
   */
  void SetBias(float* bias, cudnnHandle_t cudnn);

  /* Forward Pass Operation:
   *  Computes the forward pass of the Linear Layer on the given input
   *  Input - Values/Inputs from the previous layer (in host memory)
   *  Output - Pointer to the result array (in host memory)
   */
  float* LinearForward(float* input);
};

#endif 
