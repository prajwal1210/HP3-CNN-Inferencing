#ifndef OPERATIONS_H // include guard
#define OPERATIONS_H

#include <cuda.h>
#include <cudnn.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>
#include <utility>


#define checkCUDNN(expression)                               \
{                                                            \
    cudnnStatus_t status = (expression);                     \
    if (status != CUDNN_STATUS_SUCCESS) {                    \
      std::cerr << "Error on line " << __LINE__ << ": "      \
                << cudnnGetErrorString(status) << std::endl; \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
}

#define checkCudaErrors(expression)                          \
{                                                            \
    cublasStatus_t status = (expression);                     \
    if (status != 0) {                                       \
      std::cerr << "Error on line " << __LINE__ << ": "      \
                << status << std::endl;                      \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
}



/*Class for Convolution Operation*/
class Conv2D
{   
public:
    int out_channels;       //Number of filters = output channels
    int in_channels;        //Number of input channels
    int h;                  //Height of the filters
    int w;                  //Width of the filters
    int batchsize;          //Batchsize of the training data
    int input_height;       //Input Height
    int input_width;        //Input Width


    int padding;            //Padding - Assumed same in x and y direction
    int stride;             //Stride - Assumed same in x and y direction
    int dilation;           //Dilation
    float* weights;         //Weights of the conv layer
    bool bias_present;      //Bool to represent whether bias is present or not
    float* bias;            //Bias of the conv layer

    cudnnTensorFormat_t data_format = CUDNN_TENSOR_NCHW;
    cudnnTensorFormat_t param_format = CUDNN_TENSOR_NCHW;

    cudnnHandle_t cudnn;                                    //CUDNN Descriptor
    cudnnTensorDescriptor_t input_descriptor;               //Input Descriptor
    cudnnTensorDescriptor_t output_descriptor;              //Output Descriptor
    cudnnFilterDescriptor_t kernel_descriptor;              //Kernel Descriptor 
    cudnnTensorDescriptor_t convbias_descriptor;
    cudnnConvolutionDescriptor_t convolution_descriptor;    //Convolution Descriptor
    cudnnConvolutionFwdAlgo_t convolution_algorithm;        //Convolution Algorithm
    size_t workspace_bytes = 0;                             //Workspace size


    /*Constructor*/
    Conv2D(int out_channels, int in_channels, int h, int w, int batchsize, int padding, int stride, int dilation, int input_height, int input_width, cudnnHandle_t cudnn);
    ~Conv2D();


    /*Weight and Bias Initializers*/
    void SetWeights(float* weights);
    void SetBias(float* bias);

    /*Get Output Dimensions*/
    std::pair<int,int> GetOutputDims();

    /*Forward Pass*/
    float* ConvForward(float* input);

protected:
    /*Create Descriptors*/
    void CreateDescriptors();


};



typedef enum{
    t_max = 0,
    t_avg
} poolType;


/*Class for Pooling Operation*/
class Pool
{
//Data is in NHWC format
public:
    poolType type;
    int in_channels;        //Number of input channels
    int batchsize;          //Batchsize of the training data
    int kernel_size_x;      //Kernel Size in X
    int kernel_size_y;      //Kernel Size in Y
    int padding;            //Padding - Assumed same in x and y direction
    int stride_x;           //Stride in X
    int stride_y;           //Stride in Y
    int input_height;       //Input Height
    int input_width;        //Input Width
    
    cudnnTensorFormat_t data_format = CUDNN_TENSOR_NCHW;

    cudnnHandle_t cudnn;                                    //CUDNN Descriptor
    cudnnTensorDescriptor_t input_descriptor;               //Input Descriptor
    cudnnTensorDescriptor_t output_descriptor;              //Output Descriptor
    cudnnPoolingDescriptor_t pooling_descriptor;            //Pool Descriptor
    cudnnPoolingMode_t mode;                                //Pooling Mode Descriptor

    
    /*Constructors*/
    Pool(int type, int batchsize, int in_channels, int input_height, int input_width, int output_height, int output_width, cudnnHandle_t cudnn);                                               //Normal Pooling - Kernel and Stride can be different in x and y
    Pool(int type, int batchsize, int in_channels, int input_height, int input_width, int kernel_size, int padding, int stride, cudnnHandle_t cudnn);                                          //Normal Pooling - Assume same size, stride, padding in x and y
    Pool(int type, int batchsize, int in_channels, int input_height, int input_width, int kernel_size_y, int kernel_size_x, int padding, int stride_y, int stride_x, cudnnHandle_t cudnn);     //Adaptive Pooling
    ~Pool();

    /*Helper Function to the Constructors*/
    void InitalizeAttributes(int type, int batchsize, int in_channels, int input_height, int input_width, int kernel_size_y, int kernel_size_x, int padding, int stride_y, int stride_x, cudnnHandle_t cudnn);


    /*Get Output Dimensions*/
    void GetOutputDims(int* out_n, int* out_h, int* out_w, int* out_c);

    /*Forward Pass*/
    float* PoolForward(float* input);


protected:
    /*Create Descriptors*/
    void CreateDescriptors();

};

typedef enum{
    t_relu = 0,
    t_sigmoid
} actType;

/*Class for Activation Operation*/
class Activation
{            
//Data is in NHWC format
public:
    actType type;
    int batchsize;          //Batchsize of the training data
    int in_channels;        //Number of input channels
    int input_height;       //Input Height
    int input_width;        //Input Width

    cudnnTensorFormat_t data_format = CUDNN_TENSOR_NCHW;

    cudnnHandle_t cudnn;                                    //CUDNN Descriptor
    cudnnTensorDescriptor_t output_descriptor;              //Output Descriptor
    cudnnActivationDescriptor_t activation_descriptor;;     //Activation Descriptor
    cudnnActivationMode_t mode;                             //Activation Mode Descriptor

    
    /*Constructors*/
    Activation(int type, int batchsize, int in_channels, int input_height, int input_width, cudnnHandle_t cudnn);                                               //Normal Pooling - Kernel and Stride can be different in x and y
    ~Activation();

    /*Get Output Dimensions*/
    void GetOutputDims(int* out_n, int* out_h, int* out_w, int* out_c);

    /*Forward Pass*/
    float* ActivationForward(float* input);


protected:
    /*Create Descriptors*/
    void CreateDescriptors();

};


/*Class for Linear Layer*/
class Linear
{            
//Data is in NHWC format
public:
    int batchsize;          //Batchsize of the training data
    int out_nodes;          //Number of output nodes
    int in_nodes;           //Number of input nodes
    float* weight;         //Weights of the layer - Out_nodes X In_nodes
    bool bias_present;      //Bool to know whether bias is present or not
    float* bias;            //Bias of the Layer - Out_nodes X 1

    cudnnTensorFormat_t data_format = CUDNN_TENSOR_NCHW;

    cudnnHandle_t cudnn;                                //CUDNN Descriptor
    cublasHandle_t cublas;                              //CUBLAS Descriptor
    cudnnTensorDescriptor_t bias_input_descriptor;      //Input Descriptor for Bias Addition
    cudnnTensorDescriptor_t bias_output_descriptor;     //Output Descriptor for Bias Addition
    cudnnTensorDescriptor_t bias_descriptor;            //Bias Descriptor for Bias Addition

    
    /*Constructors*/
    Linear(int batchsize, int out_nodes, int in_nodes, cublasHandle_t cublas);
    ~Linear();

    /*Get Output Dimensions*/
    void GetOutputDims(int* out_n, int* out_h, int* out_w, int* out_c);

    /*Weight and Bias Initializers*/
    void SetWeights(float* weights);
    void SetBias(float* bias, cudnnHandle_t cudnn);

    /*Forward Pass*/
    float* LinearForward(float* input);

};



#endif 
