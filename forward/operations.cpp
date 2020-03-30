#include "operations.h"
#include <utility>


#include <iostream>

using namespace std;


Conv2D::Conv2D(int out_channels, int in_channels, int h, int w, int batchsize, int padding, int stride, int dilation, int input_height, int input_width, cudnnHandle_t cudnn){
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

Conv2D::~Conv2D(){

    cudnnDestroyTensorDescriptor(this->input_descriptor);
    cudnnDestroyTensorDescriptor(this->output_descriptor);
    cudnnDestroyFilterDescriptor(this->kernel_descriptor);
    cudnnDestroyConvolutionDescriptor(this->convolution_descriptor);
    if(this->bias_present)
        cudnnDestroyTensorDescriptor(this->convbias_descriptor);    

}

void Conv2D::CreateDescriptors(){

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
                                      /*format=*/this->param_format,                     //CHECK THIS IF THE PROTO HAS CHW
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


    pair<int,int> out_dim = this->GetOutputDims();
    checkCUDNN(cudnnCreateTensorDescriptor(&(this->output_descriptor)));
    checkCUDNN(cudnnSetTensor4dDescriptor(this->output_descriptor,
                                      /*format=*/this->data_format,
                                      /*dataType=*/CUDNN_DATA_FLOAT,
                                      /*batch_size=*/this->batchsize,
                                      /*channels=*/this->out_channels,
                                      /*image_height=*/out_dim.first,
                                      /*image_width=*/out_dim.second));
    

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

void Conv2D::SetWeights(float* weights){
    this->weights = weights;
}

void Conv2D::SetBias(float* bias){
    this->bias_present = true;
    this->bias = bias;

    checkCUDNN(cudnnCreateTensorDescriptor(&(this->convbias_descriptor)));
    checkCUDNN(cudnnSetTensor4dDescriptor(this->convbias_descriptor,
                                      /*format=*/this->param_format,                 //CHECK THIS IF THE PROTO HAS CHW
                                      /*dataType=*/CUDNN_DATA_FLOAT,
                                      /*batch_size=*/1,
                                      /*channels=*/this->out_channels,
                                      /*image_height=*/1,
                                      /*image_width=*/1));

}


pair<int,int> Conv2D::GetOutputDims(){
    int n, out_channels, out_h, out_w; 
    checkCUDNN(cudnnGetConvolution2dForwardOutputDim(this->convolution_descriptor,
                                       this->input_descriptor,
                                       this->kernel_descriptor,
                                       /*out batch size =*/&n,
                                       /*output channels =*/&out_channels,
                                       /*output height=*/&out_h,
                                       /*output width=*/&out_w ));
    
    pair<int,int> out_dim(out_h,out_w);

    return out_dim;
}

 
float* Conv2D::ConvForward(float* input){

    
    cout << "Workspace size: " << (this->workspace_bytes / 1048576.0) << "MB" << endl;

    void* d_workspace{nullptr};
    cudaMalloc(&d_workspace, this->workspace_bytes);


    int image_in_bytes = this->batchsize * this->in_channels * this->input_height * this->input_width * sizeof(float);
    pair<int,int> out_dim = this->GetOutputDims();
    int image_out_bytes = this->batchsize * this->out_channels * out_dim.first * out_dim.second * sizeof(float);
    
    cout << "Input - ( " << batchsize << ", " << this->in_channels << ", " << this->input_height << ", " << this->input_width << " )" << endl;
    cout << "Output - ( " << batchsize << ", " << this->out_channels << ", " << out_dim.first << ", " << out_dim.second << " )" << endl;

    float* d_input{nullptr};
    cudaMalloc(&d_input, image_in_bytes);
    cudaMemcpy(d_input, input, image_in_bytes, cudaMemcpyHostToDevice);


    float* d_output{nullptr};
    cudaMalloc(&d_output, image_out_bytes);
    cudaMemset(d_output, 0, image_out_bytes);


    int kernel_size = this->out_channels * this->in_channels * this->h * this->w * sizeof(float);


    float* d_kernel{nullptr};
    cudaMalloc(&d_kernel, kernel_size);
    cudaMemcpy(d_kernel, this->weights, kernel_size, cudaMemcpyHostToDevice);


    const float alpha = 1, beta = 0;
    checkCUDNN(cudnnConvolutionForward(this->cudnn,
                                   &alpha,
                                   this->input_descriptor,
                                   d_input,
                                   this->kernel_descriptor,
                                   d_kernel,
                                   this->convolution_descriptor,
                                   this->convolution_algorithm,
                                   d_workspace,
                                   this->workspace_bytes,
                                   &beta,
                                   this->output_descriptor,
                                   d_output));

    float* d_bias{nullptr};
    if(this->bias_present){

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

    //Free the temporary memory
    cudaFree(d_kernel);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_workspace);
    if(this->bias_present){
        cudaFree(d_bias);
    }
 
    return h_output;

}




Pool::Pool(int type, int batchsize, int in_channels, int input_height, int input_width, int kernel_size_y, int kernel_size_x, int padding, int stride_y, int stride_x, cudnnHandle_t cudnn){
    this->InitalizeAttributes(type, batchsize, in_channels, input_height, input_width, kernel_size_y, kernel_size_x, padding, stride_y, stride_x, cudnn);

}

Pool::Pool(int type, int batchsize, int in_channels, int input_height, int input_width, int kernel_size, int padding, int stride, cudnnHandle_t cudnn){
    this->kernel_size_x = kernel_size;
    this->kernel_size_y = kernel_size;
    this->stride_x = stride;
    this->stride_y = stride;

    this->InitalizeAttributes(type, batchsize, in_channels, input_height, input_width, kernel_size_y, kernel_size_x, padding, stride_y, stride_x, cudnn);
}

Pool::Pool(int type, int batchsize, int in_channels, int input_height, int input_width, int output_height, int output_width, cudnnHandle_t cudnn){

    //Calculate the size of the kernel and stride based on output
    int stride_y = (input_height/output_height);
    int kernel_size_y = input_height - ((output_height-1)*stride_y);
    int stride_x = (input_width/output_width);
    int kernel_size_x = input_width - ((output_width-1)*stride_x);


    this->InitalizeAttributes(type, batchsize, in_channels, input_height, input_width, kernel_size_y, kernel_size_x, 0, stride_y, stride_x, cudnn);    
}

Pool::~Pool(){
    cudnnDestroyTensorDescriptor(this->input_descriptor);
    cudnnDestroyTensorDescriptor(this->output_descriptor);
    cudnnDestroyPoolingDescriptor(this->pooling_descriptor);
}

void Pool::InitalizeAttributes(int type, int batchsize, int in_channels, int input_height, int input_width, int kernel_size_y, int kernel_size_x, int padding, int stride_y, int stride_x, cudnnHandle_t cudnn){

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

    if(this->type == t_max)
        this->mode = CUDNN_POOLING_MAX;
    else
        this->mode = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
    
    this->CreateDescriptors();

        //Check the Kernel Size - Known Issue in CUDNN
    if(this->kernel_size_y * this->kernel_size_x > 256)
    {
        cerr << "Pooling Kernel Size > 256. This is a known issue in cuDNN and so will not work..Exiting.." << endl;
        this->~Pool();
        exit(1); 

    }
}

void Pool::CreateDescriptors(){
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
    
    int out_n, out_h, out_w, out_c;
    GetOutputDims(&out_n, &out_h, &out_w, &out_c);
    checkCUDNN(cudnnCreateTensorDescriptor(&(this->output_descriptor)));
    checkCUDNN(cudnnSetTensor4dDescriptor(this->output_descriptor,
                                      /*format=*/this->data_format,
                                      /*dataType=*/CUDNN_DATA_FLOAT,
                                      /*batch_size=*/this->batchsize,
                                      /*channels=*/this->in_channels,
                                      /*image_height=*/out_h,
                                      /*image_width=*/out_w));


}

void Pool::GetOutputDims(int* out_n, int* out_h, int* out_w, int* out_c){
    checkCUDNN(cudnnGetPooling2dForwardOutputDim(this->pooling_descriptor,
                                       this->input_descriptor,
                                       /*out batch size =*/out_n,
                                       /*output channels =*/out_c,
                                       /*output height=*/out_h,
                                       /*output width=*/out_w ));

}

float* Pool::PoolForward(float* input){

    int image_in_bytes = this->batchsize * this->in_channels * this->input_height * this->input_width * sizeof(float);
    int out_n, out_h, out_w, out_c;
    GetOutputDims(&out_n, &out_h, &out_w, &out_c);
    int image_out_bytes = out_n * out_c * out_h * out_w * sizeof(float);
    
    cout << "Input - ( " << this->batchsize << ", " << this->in_channels << ", " << this->input_height << ", " << this->input_width << " )" << endl;
    cout << "Output - ( " << out_n << ", " << out_c << ", " << out_h << ", " << out_w << " )" << endl;
    cout << "Kernel Size - ( " << this->kernel_size_y << ", " << this->kernel_size_x << " )" << endl;
    cout << "Stride - ( " << this->stride_y << ", " << this->stride_x << " )" << endl;

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

    //Free the temporary memory
    cudaFree(d_input);
    cudaFree(d_output);
 
    return h_output;
}



Activation::Activation(int type, int batchsize, int in_channels, int input_height, int input_width, cudnnHandle_t cudnn){
    this->type = (actType)type;
    this->batchsize = batchsize;
    this->in_channels = in_channels;
    this->input_height = input_height;
    this->input_width = input_width;
    this->cudnn = cudnn;

    if(this->type == t_relu)
        this->mode = CUDNN_ACTIVATION_RELU;
    else
        this->mode = CUDNN_ACTIVATION_SIGMOID;

    this->CreateDescriptors();
}

Activation::~Activation(){
    cudnnDestroyTensorDescriptor(this->output_descriptor);
    cudnnDestroyActivationDescriptor(this->activation_descriptor);
}

void Activation::CreateDescriptors(){

    //Input Dimensions = Output Dimensions
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

void Activation::GetOutputDims(int* out_n, int* out_h, int* out_w, int* out_c){
    *out_n = this->batchsize; 
    *out_h = this->input_height;
    *out_w = this->input_width;
    *out_c = this->in_channels;
}

float* Activation::ActivationForward(float* input){
        
    int image_bytes = this->batchsize * this->in_channels * this->input_height * this->input_width * sizeof(float);
    
    cout << "Input - ( " << this->batchsize << ", " << this->in_channels << ", " << this->input_height << ", " << this->input_width << " )" << endl;
    cout << "Output - ( " << this->batchsize << ", " << this->in_channels << ", " << this->input_height << ", " << this->input_width << " )" << endl;


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

    //Free the temporary memory
    cudaFree(d_output);
 
    return h_output;

}


Linear::Linear(int batchsize, int out_nodes, int in_nodes, cublasHandle_t cublas){
    this->batchsize = batchsize;
    this->out_nodes = out_nodes;
    this->in_nodes = in_nodes;
    this->bias_present = false;
    this->cublas = cublas;
}

Linear::~Linear(){
    if(this->bias_present){
        cudnnDestroyTensorDescriptor(this->bias_input_descriptor);
        cudnnDestroyTensorDescriptor(this->bias_output_descriptor);
        cudnnDestroyTensorDescriptor(this->bias_descriptor);
    }
}

void Linear::SetWeights(float* weights){
    this->weight = weights;
}

void Linear::SetBias(float* bias, cudnnHandle_t cudnn){
    this->bias_present = true;
    this->bias = bias;

    this->cudnn = cudnn;

    //Bias Input Descriptor
    checkCUDNN(cudnnCreateTensorDescriptor(&(this->bias_input_descriptor)));
    checkCUDNN(cudnnSetTensor4dDescriptor(this->bias_input_descriptor,
                                      /*format=*/this->data_format,
                                      /*dataType=*/CUDNN_DATA_FLOAT,
                                      /*batch_size=*/this->batchsize,
                                      /*channels=*/this->out_nodes,
                                      /*image_height=*/1,
                                      /*image_width=*/1));

    //Bias Output Descriptor
    checkCUDNN(cudnnCreateTensorDescriptor(&(this->bias_output_descriptor)));
    checkCUDNN(cudnnSetTensor4dDescriptor(this->bias_output_descriptor,
                                      /*format=*/this->data_format,
                                      /*dataType=*/CUDNN_DATA_FLOAT,
                                      /*batch_size=*/this->batchsize,
                                      /*channels=*/this->out_nodes,
                                      /*image_height=*/1,
                                      /*image_width=*/1));


    //Create Bias Descriptor
    checkCUDNN(cudnnCreateTensorDescriptor(&(this->bias_descriptor)));
    checkCUDNN(cudnnSetTensor4dDescriptor(this->bias_descriptor,
                                      /*format=*/this->data_format,                 //CHECK THIS IF THE PROTO HAS CHW
                                      /*dataType=*/CUDNN_DATA_FLOAT,
                                      /*batch_size=*/1,
                                      /*channels=*/this->out_nodes,
                                      /*image_height=*/1,
                                      /*image_width=*/1));

}

void Linear::GetOutputDims(int* n, int* c, int* h, int* w){
    *n = this->batchsize;
    *c = this->out_nodes;
    *h = 1;
    *w = 1;
}

float* Linear::LinearForward(float* input){
    int input_bytes = this->batchsize * this->in_nodes * sizeof(float);
    int output_bytes = this->batchsize * this->out_nodes * sizeof(float);
    int weight_bytes = this->out_nodes * this->in_nodes * sizeof(float);

    cout << "Input - ( " << this->batchsize << ", " << this->in_nodes << ", " << 1 << ", " << 1 << " )" << endl;
    cout << "Output - ( " << this->batchsize << ", " << this->out_nodes << ", " << 1 << ", " << 1 << " )" << endl;

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
    checkCudaErrors(cublasSgemm(this->cublas,                                       //CUBLAS works in column major form              
                                CUBLAS_OP_T, CUBLAS_OP_N,                           //Weights[Out_nodes X In_nodes] in C fashion (row major) => Weights[In_Nodes X Out_Nodes] in CUBLAS input
                                this->out_nodes, this->batchsize, this->in_nodes,   //Input[Batchsize X In_nodes] in C fashion (row major) => Input[In_Nodes X Batchize] in CUBLAS input
                                &alpha,                                             //Output[Out_nodes X Batchsize] in CUBLAS = Weights[In_Nodes X Out_Nodes]^T * Input[In_Nodes X Batchize]
                                d_weight, this->in_nodes,                           //Therefore, m = Out_nodes, n = Batch_Size, k = In_nodes (Weights^T)
                                d_input, this->in_nodes,                            //LDA = In_nodes (Weights), LDB = In_Nodes (Input), LDC = Out_Nodes
                                &beta,                                              //Output[Out_nodes X Batchsize] in CUBLAS => Output[Batchsize X Out_nodes] in C
                                d_output, this->out_nodes));
    
    int bias_bytes = this->out_nodes * sizeof(float);
    float* d_bias{nullptr};

    if(this->bias_present){
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

    //Free the temporary memory
    cudaFree(d_weight);
    cudaFree(d_input);
    cudaFree(d_output);
    if(this->bias_present){
        cudaFree(d_bias);
    }
 
    return h_output;

}