/************************************************************
 * operations.cc:									                          *
 * Implementation of the library operations.h               *
 *                                                          *
 * Author: Prajwal Singhania								                *
 ************************************************************/

#include "operations.h"

/* (Conv2D)Constructor Implementation */
Conv2D::Conv2D(int out_channels, int in_channels, int h, int w, int batchsize, int padding, 
              int stride, int dilation, int input_height, int input_width, cudnnHandle_t cudnn) {
  this->out_channels = out_channels;
  this->in_channels = in_channels;
  this->h = h;
  this->w = w;
  this->batchsize = batchsize;
  this->padding = padding;
  this->stride = stride;
  this->dilation = dilation;
  this->cudnn = cudnn;
  this->bias_present = false;
  this->input_height = input_height;
  this->input_width = input_width;
  this->CreateDescriptors();
}

/* (Conv2D)Destructor Implementation */
Conv2D::~Conv2D() {
  cudnnDestroyTensorDescriptor(this->input_descriptor);
  cudnnDestroyTensorDescriptor(this->output_descriptor);
  cudnnDestroyFilterDescriptor(this->kernel_descriptor);
  cudnnDestroyConvolutionDescriptor(this->convolution_descriptor);
  if (this->bias_present)
    cudnnDestroyTensorDescriptor(this->convbias_descriptor);    
}

/* (Conv2D)SetWeights Implementation */
void Conv2D::SetWeights(float* weights) {
  this->weights = weights;
}

/* (Conv2D)SetBias Implementation : Also creates bias specific descriptors */
void Conv2D::SetBias(float* bias) {
  this->bias_present = true;
  this->bias = bias;

  checkCUDNN(cudnnCreateTensorDescriptor(&(this->convbias_descriptor)));
  checkCUDNN(cudnnSetTensor4dDescriptor(this->convbias_descriptor,
                                        /*format=*/this->param_format,
                                        /*dataType=*/CUDNN_DATA_FLOAT,
                                        /*batch_size=*/1,
                                        /*channels=*/this->out_channels,
                                        /*image_height=*/1,
                                        /*image_width=*/1));
}

/* (Conv2D)GetOutputDims Implementation */
void Conv2D::GetOutputDims(int* out_n, int* out_c, int* out_h, int* out_w) {
  checkCUDNN(cudnnGetConvolution2dForwardOutputDim(this->convolution_descriptor,
                                                  this->input_descriptor,
                                                  this->kernel_descriptor,
                                                  /*out batch size =*/out_n,
                                                  /*output channels =*/out_c,
                                                  /*output height=*/out_h,
                                                  /*output width=*/out_w ));
}

/* (Conv2D)ConvForward Implementation : Uses CUDNN function call */
float* Conv2D::ConvForward(float* input) {
  return this->Conv_Direct(input);
//   std::cout << "Workspace size: " << (this->workspace_bytes / 1048576.0) << "MB" << std::endl;

//   void* d_workspace{nullptr};
//   cudaMalloc(&d_workspace, this->workspace_bytes);


//   int image_in_bytes = this->batchsize * this->in_channels * this->input_height * this->input_width * sizeof(float);
//   int out_n, out_c, out_h, out_w;
//   this->GetOutputDims(&out_n, &out_c, &out_h, &out_w);
//   int image_out_bytes = this->batchsize * this->out_channels * out_h * out_w * sizeof(float);
  
//   std::cout << "Input - ( " << batchsize << ", " << this->in_channels << ", " << this->input_height << ", " << this->input_width << " )" << std::endl;
//   std::cout << "Output - ( " << batchsize << ", " << this->out_channels << ", " << out_h << ", " << out_w << " )" << std::endl;

//   float* d_input{nullptr};
//   cudaMalloc(&d_input, image_in_bytes);
//   cudaMemcpy(d_input, input, image_in_bytes, cudaMemcpyHostToDevice);


//   float* d_output{nullptr};
//   cudaMalloc(&d_output, image_out_bytes);
//   cudaMemset(d_output, 0, image_out_bytes);


//   int kernel_size = this->out_channels * this->in_channels * this->h * this->w * sizeof(float);


//   float* d_kernel{nullptr};
//   cudaMalloc(&d_kernel, kernel_size);
//   cudaMemcpy(d_kernel, this->weights, kernel_size, cudaMemcpyHostToDevice);


//   const float alpha = 1, beta = 0;
//   checkCUDNN(cudnnConvolutionForward(this->cudnn,
//                                     &alpha,
//                                     this->input_descriptor,
//                                     d_input,
//                                     this->kernel_descriptor,
//                                     d_kernel,
//                                     this->convolution_descriptor,
//                                     this->convolution_algorithm,
//                                     d_workspace,
//                                     this->workspace_bytes,
//                                     &beta,
//                                     this->output_descriptor,
//                                     d_output));

//   float* d_bias{nullptr};
//   if (this->bias_present) {
//     int bias_size = this->out_channels * sizeof(float);
//     cudaMalloc(&d_bias, bias_size);
//     cudaMemcpy(d_bias, this->bias, bias_size, cudaMemcpyHostToDevice);
    
//     checkCUDNN(cudnnAddTensor(this->cudnn, 
//                               &alpha,
//                               this->convbias_descriptor,
//                               d_bias, 
//                               &alpha,
//                               this->output_descriptor, 
//                               d_output));
//   }
  
//   float* h_output = new float[image_out_bytes];
//   cudaMemcpy(h_output, d_output, image_out_bytes, cudaMemcpyDeviceToHost);

//   /* Free the temporary memory */
//   cudaFree(d_kernel);
//   cudaFree(d_input);
//   cudaFree(d_output);
//   cudaFree(d_workspace);
//   if (this->bias_present) cudaFree(d_bias);

//   return h_output;
}

float* Conv2D::Conv_Direct(float* input) {
  int image_in_bytes = this->batchsize * this->in_channels * this->input_height * this->input_width * sizeof(float);
  
  std::cout << "Input - ( " << this->batchsize << ", " << this->in_channels << ", " << this->input_height << ", " << this->input_width << " )" << std::endl;

  float* d_input{nullptr};
  cudaMalloc(&d_input, image_in_bytes);
  cudaMemcpy(d_input, input, image_in_bytes, cudaMemcpyHostToDevice);

  int kernel_size = this->out_channels * this->in_channels * this->h * this->w * sizeof(float);

  float* d_kernel{nullptr};
  cudaMalloc(&d_kernel, kernel_size);
  cudaMemcpy(d_kernel, this->weights, kernel_size, cudaMemcpyHostToDevice);


  float* d_output =  Direct::passforward(this->out_channels, this->in_channels, this->h, this->w, this->padding, this->stride, 
                                    d_kernel, this->batchsize, this->input_height, this->input_width, d_input);
  
  int out_n, out_c, out_h, out_w;
  this->GetOutputDims(&out_n, &out_c, &out_h, &out_w);
  int image_out_bytes = this->batchsize * this->out_channels * out_h * out_w * sizeof(float);
  
  std::cout << "Output - ( " << this->batchsize << ", " << this->out_channels << ", " << out_h << ", " << out_w << " )" << std::endl;

  const float alpha = 1, beta = 0;
  float* d_bias{nullptr};
  if (this->bias_present) {
    int bias_size = this->out_channels * sizeof(float);
    cudaMalloc(&d_bias, bias_size);
    cudaMemcpy(d_bias, this->bias, bias_size, cudaMemcpyHostToDevice);
    
    checkCUDNN(cudnnAddTensor(this->cudnn, 
                              &alpha,
                              this->convbias_descriptor,
                              d_bias, 
                              &alpha,
                              this->output_descriptor, 
                              d_output));
  }

  float* h_output = new float[image_out_bytes];
  cudaMemcpy(h_output, d_output, image_out_bytes, cudaMemcpyDeviceToHost);
  
  /* Free the temporary memory */
  cudaFree(d_input);
  cudaFree(d_kernel);
  cudaFree(d_output);
  if (this->bias_present) cudaFree(d_bias);

  return h_output;
}

/* (Conv2D)CreateDescriptors Implementation */
void Conv2D::CreateDescriptors() {
  checkCUDNN(cudnnCreateTensorDescriptor(&(this->input_descriptor)));
  checkCUDNN(cudnnSetTensor4dDescriptor(this->input_descriptor,
                                        /*format=*/this->data_format,
                                        /*dataType=*/CUDNN_DATA_FLOAT,
                                        /*batch_size=*/this->batchsize,
                                        /*channels=*/this->in_channels,
                                        /*image_height=*/this->input_height,
                                        /*image_width=*/this->input_width));

  checkCUDNN(cudnnCreateFilterDescriptor(&(this->kernel_descriptor)));
  checkCUDNN(cudnnSetFilter4dDescriptor(this->kernel_descriptor,
                                        /*dataType=*/CUDNN_DATA_FLOAT,
                                        /*format=*/this->param_format,
                                        /*out_channels=*/this->out_channels,
                                        /*in_channels=*/this->in_channels,
                                        /*kernel_height=*/this->h,
                                        /*kernel_width=*/this->w));
  
  checkCUDNN(cudnnCreateConvolutionDescriptor(&(this->convolution_descriptor)));
  checkCUDNN(cudnnSetConvolution2dDescriptor(this->convolution_descriptor,
                                            /*pad_height=*/this->padding,
                                            /*pad_width=*/this->padding,
                                            /*vertical_stride=*/this->stride,
                                            /*horizontal_stride=*/this->stride,
                                            /*dilation_height=*/this->dilation,
                                            /*dilation_width=*/this->dilation,
                                            /*mode=*/CUDNN_CROSS_CORRELATION,
                                            /*computeType=*/CUDNN_DATA_FLOAT));

  int out_n, out_c, out_h, out_w;
  this->GetOutputDims(&out_n, &out_c, &out_h, &out_w);
  checkCUDNN(cudnnCreateTensorDescriptor(&(this->output_descriptor)));
  checkCUDNN(cudnnSetTensor4dDescriptor(this->output_descriptor,
                                        /*format=*/this->data_format,
                                        /*dataType=*/CUDNN_DATA_FLOAT,
                                        /*batch_size=*/this->batchsize,
                                        /*channels=*/this->out_channels,
                                        /*image_height=*/out_h,
                                        /*image_width=*/out_w));
  

  checkCUDNN(cudnnGetConvolutionForwardAlgorithm(this->cudnn,
                                                this->input_descriptor,
                                                this->kernel_descriptor,
                                                this->convolution_descriptor,
                                                this->output_descriptor,
                                                CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                                /*memoryLimitInBytes=*/0,
                                                &(this->convolution_algorithm)));
  
  this->workspace_bytes = 0;
  checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(this->cudnn,
                                                    this->input_descriptor,
                                                    this->kernel_descriptor,
                                                    this->convolution_descriptor,
                                                    this->output_descriptor,
                                                    this->convolution_algorithm,
                                                    &(this->workspace_bytes)));
}



/* (Pool)Normal Pooling Constructor Implementation : Kernel Size and Stride can be different in X and Y direction */
Pool::Pool(int type, int batchsize, int in_channels, int input_height, int input_width, 
          int kernel_size_y, int kernel_size_x, int padding, int stride_y, int stride_x, cudnnHandle_t cudnn) {
  this->InitalizeAttributes(type, batchsize, in_channels, input_height, input_width, 
                            kernel_size_y, kernel_size_x, padding, stride_y, stride_x, cudnn);
}

/* (Pool)Normal Pooling Constructor Implementation : Kernel Size and Stride assumed same in X and Y direction */
Pool::Pool(int type, int batchsize, int in_channels, int input_height, int input_width, int kernel_size, 
          int padding, int stride, cudnnHandle_t cudnn) {
  this->kernel_size_x = kernel_size;
  this->kernel_size_y = kernel_size;
  this->stride_x = stride;
  this->stride_y = stride;

  this->InitalizeAttributes(type, batchsize, in_channels, input_height, input_width, 
                            kernel_size_y, kernel_size_x, padding, stride_y, stride_x, cudnn);
}

/* (Pool)Adaptive Pooling Constructor Implementation */
Pool::Pool(int type, int batchsize, int in_channels, int input_height, int input_width, 
          int output_height, int output_width, cudnnHandle_t cudnn) {
  /*Calculate the size of the kernel and stride based on output*/
  int stride_y = (input_height/output_height);
  int kernel_size_y = input_height - ((output_height-1)*stride_y);
  int stride_x = (input_width/output_width);
  int kernel_size_x = input_width - ((output_width-1)*stride_x);

  this->InitalizeAttributes(type, batchsize, in_channels, input_height, input_width, 
                            kernel_size_y, kernel_size_x, 0, stride_y, stride_x, cudnn);    
}

/* (Pool)Destructor Implementation */
Pool::~Pool() {
  cudnnDestroyTensorDescriptor(this->input_descriptor);
  cudnnDestroyTensorDescriptor(this->output_descriptor);
  cudnnDestroyPoolingDescriptor(this->pooling_descriptor);
}

/* (Pool)GetOutputDims Implementation */
void Pool::GetOutputDims(int* out_n, int* out_c, int* out_h, int* out_w) {
  checkCUDNN(cudnnGetPooling2dForwardOutputDim(this->pooling_descriptor,
                                              this->input_descriptor,
                                              /*out batch size =*/out_n,
                                              /*output channels =*/out_c,
                                              /*output height=*/out_h,
                                              /*output width=*/out_w ));

}

/* (Pool)PoolForward Implementation : Uses CUDNN function call */
float* Pool::PoolForward(float* input) {
  int image_in_bytes = this->batchsize * this->in_channels * this->input_height * this->input_width * sizeof(float);
  int out_n, out_c, out_h, out_w;
  GetOutputDims(&out_n, &out_c, &out_h, &out_w);
  int image_out_bytes = out_n * out_c * out_h * out_w * sizeof(float);
  
  std::cout << "Input - ( " << this->batchsize << ", " << this->in_channels << ", " << this->input_height << ", " << this->input_width << " )" << std::endl;
  std::cout << "Output - ( " << out_n << ", " << out_c << ", " << out_h << ", " << out_w << " )" << std::endl;
  std::cout << "Kernel Size - ( " << this->kernel_size_y << ", " << this->kernel_size_x << " )" << std::endl;
  std::cout << "Stride - ( " << this->stride_y << ", " << this->stride_x << " )" << std::endl;

  float* d_input{nullptr};
  cudaMalloc(&d_input, image_in_bytes);
  cudaMemcpy(d_input, input, image_in_bytes, cudaMemcpyHostToDevice);


  float* d_output{nullptr};
  cudaMalloc(&d_output, image_out_bytes);
  cudaMemset(d_output, 0, image_out_bytes);

  const float alpha = 1, beta = 0;
  checkCUDNN(cudnnPoolingForward(this->cudnn,
                                this->pooling_descriptor,
                                &alpha,
                                this->input_descriptor,
                                d_input,
                                &beta,
                                this->output_descriptor,
                                d_output));

  float* h_output = new float[image_out_bytes];
  cudaMemcpy(h_output, d_output, image_out_bytes, cudaMemcpyDeviceToHost);

  /* Free the temporary memory */
  cudaFree(d_input);
  cudaFree(d_output);

  return h_output;
}

/* (Pool)InitalizeAttributes Implementation : Helper to the constructor to initialize data members */
void Pool::InitalizeAttributes(int type, int batchsize, int in_channels, int input_height, int input_width, 
                              int kernel_size_y, int kernel_size_x, int padding, int stride_y, int stride_x, cudnnHandle_t cudnn) {
  this->type = (poolType)type;
  this->kernel_size_x = kernel_size_x;
  this->kernel_size_y = kernel_size_y;
  this->batchsize = batchsize;
  this->in_channels = in_channels;
  this->padding = padding;
  this->stride_x = stride_x;
  this->stride_y = stride_y;
  this->input_height = input_height;
  this->input_width = input_width;
  this->cudnn = cudnn;

  if (this->type == t_max)
    this->mode = CUDNN_POOLING_MAX;
  else
    this->mode = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
  
  this->CreateDescriptors();

  /* Check the Kernel Size - Known Issue in CUDNN */
  if (this->kernel_size_y * this->kernel_size_x > 256) {
    std::cerr << "Pooling Kernel Size > 256. This is a known issue in cuDNN and so will not work..Exiting.." << std::endl;
    this->~Pool();
    exit(1); 
  }
}

/* (Pool)CreateDescriptors Implementation */
void Pool::CreateDescriptors() {
  checkCUDNN(cudnnCreateTensorDescriptor(&(this->input_descriptor)));
  checkCUDNN(cudnnSetTensor4dDescriptor(this->input_descriptor,
                                        /*format=*/this->data_format,
                                        /*dataType=*/CUDNN_DATA_FLOAT,
                                        /*batch_size=*/this->batchsize,
                                        /*channels=*/this->in_channels,
                                        /*image_height=*/this->input_height,
                                        /*image_width=*/this->input_width));

  checkCUDNN(cudnnCreatePoolingDescriptor(&(this->pooling_descriptor)));
  checkCUDNN(cudnnSetPooling2dDescriptor(this->pooling_descriptor,
                                        this->mode,
                                        CUDNN_PROPAGATE_NAN,
                                        /*windowHeight*/this->kernel_size_y,
                                        /*windowWidth*/this->kernel_size_x,
                                        /*verticalPadding*/this->padding,
                                        /*horizontalPadding*/this->padding,
                                        /*verticalStride*/this->stride_y,
                                        /*horizontalStride*/this->stride_x));
  
  int out_n, out_c, out_h, out_w;
  GetOutputDims(&out_n, &out_c, &out_h, &out_w);
  checkCUDNN(cudnnCreateTensorDescriptor(&(this->output_descriptor)));
  checkCUDNN(cudnnSetTensor4dDescriptor(this->output_descriptor,
                                        /*format=*/this->data_format,
                                        /*dataType=*/CUDNN_DATA_FLOAT,
                                        /*batch_size=*/this->batchsize,
                                        /*channels=*/this->in_channels,
                                        /*image_height=*/out_h,
                                        /*image_width=*/out_w));
}



/* (Activation)Constructor Implementation */
Activation::Activation(int type, int batchsize, int in_channels, int input_height, int input_width, cudnnHandle_t cudnn) {
  this->type = (actType)type;
  this->batchsize = batchsize;
  this->in_channels = in_channels;
  this->input_height = input_height;
  this->input_width = input_width;
  this->cudnn = cudnn;

  if (this->type == t_relu)
    this->mode = CUDNN_ACTIVATION_RELU;
  else
    this->mode = CUDNN_ACTIVATION_SIGMOID;

  this->CreateDescriptors();
}

/* (Activation)Destructor Implementation */
Activation::~Activation() {
  cudnnDestroyTensorDescriptor(this->output_descriptor);
  cudnnDestroyActivationDescriptor(this->activation_descriptor);
}

/* (Activation)GetOutputDims Implementation */
void Activation::GetOutputDims(int* out_n, int* out_c, int* out_h, int* out_w) {
  *out_n = this->batchsize; 
  *out_c = this->in_channels;
  *out_h = this->input_height;
  *out_w = this->input_width;
}

/* (Activation)ActivationForward Implementation : Uses CUDNN function call */
float* Activation::ActivationForward(float* input) { 
  int image_bytes = this->batchsize * this->in_channels * this->input_height * this->input_width * sizeof(float);
  
  std::cout << "Input - ( " << this->batchsize << ", " << this->in_channels << ", " << this->input_height << ", " << this->input_width << " )" << std::endl;
  std::cout << "Output - ( " << this->batchsize << ", " << this->in_channels << ", " << this->input_height << ", " << this->input_width << " )" << std::endl;


  float* d_output{nullptr};
  cudaMalloc(&d_output, image_bytes);
  cudaMemcpy(d_output, input, image_bytes, cudaMemcpyHostToDevice);

  const float alpha = 1, beta = 0;
  checkCUDNN(cudnnActivationForward(this->cudnn,
                                    this->activation_descriptor,
                                    &alpha,
                                    this->output_descriptor,
                                    d_output,
                                    &beta,
                                    this->output_descriptor,
                                    d_output));


  float* h_output = new float[image_bytes];
  cudaMemcpy(h_output, d_output, image_bytes, cudaMemcpyDeviceToHost);

  /* Free the temporary memory */
  cudaFree(d_output);

  return h_output;
}

/* (Activation)CreateDescriptors Implementation */
void Activation::CreateDescriptors() {
  /* Input Dimensions = Output Dimensions */
  checkCUDNN(cudnnCreateTensorDescriptor(&(this->output_descriptor)));
  checkCUDNN(cudnnSetTensor4dDescriptor(this->output_descriptor,
                                        /*format=*/this->data_format,
                                        /*dataType=*/CUDNN_DATA_FLOAT,
                                        /*batch_size=*/this->batchsize,
                                        /*channels=*/this->in_channels,
                                        /*image_height=*/this->input_height,
                                        /*image_width=*/this->input_width));


  checkCUDNN(cudnnCreateActivationDescriptor(&(this->activation_descriptor)));
  checkCUDNN(cudnnSetActivationDescriptor(this->activation_descriptor,
                                          /*mode=*/this->mode,
                                          /*reluNanOpt=*/CUDNN_PROPAGATE_NAN,
                                          /*relu_coef=*/0));                               
}



/* (Linear)Constructor Implementation */
Linear::Linear(int batchsize, int out_nodes, int in_nodes, cublasHandle_t cublas) {
  this->batchsize = batchsize;
  this->out_nodes = out_nodes;
  this->in_nodes = in_nodes;
  this->bias_present = false;
  this->cublas = cublas;
}

/* (Linear)Destructor Implementation */
Linear::~Linear() {
  if (this->bias_present) {
    cudnnDestroyTensorDescriptor(this->bias_input_descriptor);
    cudnnDestroyTensorDescriptor(this->bias_output_descriptor);
    cudnnDestroyTensorDescriptor(this->bias_descriptor);
  }
}

/* (Linear)SetWeights Implementation */
void Linear::SetWeights(float* weights) {
  this->weight = weights;
}

/* (Linear)SetBias Implementation : Also creates bias specific descriptors */
void Linear::SetBias(float* bias, cudnnHandle_t cudnn) {
  this->bias_present = true;
  this->bias = bias;
  this->cudnn = cudnn;

  /* Bias Input Descriptor */
  checkCUDNN(cudnnCreateTensorDescriptor(&(this->bias_input_descriptor)));
  checkCUDNN(cudnnSetTensor4dDescriptor(this->bias_input_descriptor,
                                        /*format=*/this->data_format,
                                        /*dataType=*/CUDNN_DATA_FLOAT,
                                        /*batch_size=*/this->batchsize,
                                        /*channels=*/this->out_nodes,
                                        /*image_height=*/1,
                                        /*image_width=*/1));

  /* Bias Output Descriptor */
  checkCUDNN(cudnnCreateTensorDescriptor(&(this->bias_output_descriptor)));
  checkCUDNN(cudnnSetTensor4dDescriptor(this->bias_output_descriptor,
                                        /*format=*/this->data_format,
                                        /*dataType=*/CUDNN_DATA_FLOAT,
                                        /*batch_size=*/this->batchsize,
                                        /*channels=*/this->out_nodes,
                                        /*image_height=*/1,
                                        /*image_width=*/1));

  /* Bias Descriptor */
  checkCUDNN(cudnnCreateTensorDescriptor(&(this->bias_descriptor)));
  checkCUDNN(cudnnSetTensor4dDescriptor(this->bias_descriptor,
                                        /*format=*/this->data_format,
                                        /*dataType=*/CUDNN_DATA_FLOAT,
                                        /*batch_size=*/1,
                                        /*channels=*/this->out_nodes,
                                        /*image_height=*/1,
                                        /*image_width=*/1));
}

/* (Linear)GetOutputDims Implementation */
void Linear::GetOutputDims(int* out_n, int* out_c, int* out_h, int* out_w) {
  *out_n = this->batchsize;
  *out_c = this->out_nodes;
  *out_h = 1;
  *out_w = 1;
}

/* (Linear)LinearForward Implementation : Uses CUBLAS GEMM function for matrix multiplication and CUDNN for bias addition */
float* Linear::LinearForward(float* input) {
  int input_bytes = this->batchsize * this->in_nodes * sizeof(float);
  int output_bytes = this->batchsize * this->out_nodes * sizeof(float);
  int weight_bytes = this->out_nodes * this->in_nodes * sizeof(float);

  std::cout << "Input - ( " << this->batchsize << ", " << this->in_nodes << ", " << 1 << ", " << 1 << " )" << std::endl;
  std::cout << "Output - ( " << this->batchsize << ", " << this->out_nodes << ", " << 1 << ", " << 1 << " )" << std::endl;

  float* d_input{nullptr};
  cudaMalloc(&d_input, input_bytes);
  cudaMemcpy(d_input, input, input_bytes, cudaMemcpyHostToDevice);

  float* d_output{nullptr};
  cudaMalloc(&d_output, output_bytes);
  cudaMemset(d_output, 0, output_bytes);

  float* d_weight{nullptr};
  cudaMalloc(&d_weight, weight_bytes);
  cudaMemcpy(d_weight, this->weight, weight_bytes, cudaMemcpyHostToDevice);


  const float alpha = 1, beta = 0;
  
  /* CUBLAS works in column major form:
   * Weights[Out_nodes X In_nodes] in C fashion (row major) => Weights[In_Nodes X Out_Nodes] in CUBLAS input
   * Input[Batchsize X In_nodes] in C fashion (row major) => Input[In_Nodes X Batchize] in CUBLAS input
   * Output[Out_nodes X Batchsize] in CUBLAS = Weights[In_Nodes X Out_Nodes]^T * Input[In_Nodes X Batchize]
   * Therefore, m = Out_nodes, n = Batch_Size, k = In_nodes (Weights^T)
   * LDA = In_nodes (Weights), LDB = In_Nodes (Input), LDC = Out_Nodes
   * Output[Out_nodes X Batchsize] in CUBLAS => Output[Batchsize X Out_nodes] in C
   */
  checkCudaErrors(cublasSgemm(this->cublas,                                                   
                              CUBLAS_OP_T, CUBLAS_OP_N,                           
                              this->out_nodes, this->batchsize, this->in_nodes,   
                              &alpha,                                             
                              d_weight, this->in_nodes,                           
                              d_input, this->in_nodes,                            
                              &beta,                                              
                              d_output, this->out_nodes));
  
  int bias_bytes = this->out_nodes * sizeof(float);
  float* d_bias{nullptr};

  if (this->bias_present) {
    int bias_bytes = this->out_nodes * sizeof(float);
    cudaMalloc(&d_bias, bias_bytes);
    cudaMemcpy(d_bias, this->bias, bias_bytes, cudaMemcpyHostToDevice);
    
    checkCUDNN(cudnnAddTensor(this->cudnn, 
                              &alpha,
                              this->bias_descriptor,
                              d_bias, 
                              &alpha,
                              this->bias_output_descriptor, 
                              d_output));
  }

  float* h_output = new float[output_bytes];
  cudaMemcpy(h_output, d_output, output_bytes, cudaMemcpyDeviceToHost);

  /* Free the temporary memory */
  cudaFree(d_weight);
  cudaFree(d_input);
  cudaFree(d_output);
  if (this->bias_present) cudaFree(d_bias);

  return h_output;

}