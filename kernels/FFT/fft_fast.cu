#include "fftheader.h"

/* CUFFT ERROR CHECK */
static const char *_cudaGetErrorEnum(cufftResult error) {
  switch (error) {
    case CUFFT_SUCCESS:
      return "CUFFT_SUCCESS";

    case CUFFT_INVALID_PLAN:
      return "CUFFT_INVALID_PLAN";

    case CUFFT_ALLOC_FAILED:
      return "CUFFT_ALLOC_FAILED";

    case CUFFT_INVALID_TYPE:
      return "CUFFT_INVALID_TYPE";

    case CUFFT_INVALID_VALUE:
      return "CUFFT_INVALID_VALUE";

    case CUFFT_INTERNAL_ERROR:
      return "CUFFT_INTERNAL_ERROR";

    case CUFFT_EXEC_FAILED:
      return "CUFFT_EXEC_FAILED";

    case CUFFT_SETUP_FAILED:
      return "CUFFT_SETUP_FAILED";

    case CUFFT_INVALID_SIZE:
      return "CUFFT_INVALID_SIZE";

    case CUFFT_UNALIGNED_DATA:
      return "CUFFT_UNALIGNED_DATA";
  }

  return "<unknown>";
}

#define cufftSafeCall(err)      __cufftSafeCall(err, __FILE__, __LINE__)

inline void __cufftSafeCall(cufftResult err, const char *file, const int line)
{
  if( CUFFT_SUCCESS != err) {
              fprintf(stderr, "CUFFT error in file '%s', line %d\nerror %d: %s\nterminating!\n",__FILE__, __LINE__,err, \
                          _cudaGetErrorEnum(err)); \
          cudaDeviceReset(); assert(0); \
  }
}

/* multiplication of 2 cufftComplex elements
 * One complex product for each thread scaled by total elements
 */
__global__ void pointwise_product(cufftComplex* d_outA, cufftComplex* d_outB, float size, float scale)
{
  int i = blockIdx.x *blockDim.x + threadIdx.x;

  if (i < size) {
    float a,b;
    a = d_outA[i].x * d_outB[i].x - d_outA[i].y * d_outB[i].y;
    b = d_outA[i].x * d_outB[i].y + d_outA[i].y * d_outB[i].x;
    d_outA[i].x = a * scale;
    d_outA[i].y = b * scale;
  }
}

/*flip filter about the center element
 * <<H,W,D>>
 */
 __global__ void flip_filer(float* f_in, float* f_out, int H, int W, int D, int out_size)
{
  int col = blockIdx.x*blockDim.x+threadIdx.x;
  int row = blockIdx.y*blockDim.y+threadIdx.y;
  int dep = blockIdx.z*blockDim.z+threadIdx.z;

  int new_col = H - col -1;
  int new_row = W - row - 1;
  int new_dep = D - dep - 1;

  if(col < H && row < W && dep < D) {
    #pragma unroll
    for(int itr = 0; itr < out_size; itr++)  
    {  
      int i = itr * D * H * W + dep * H * W + col * W + row ;
      int j = itr * D * H * W + new_dep * H * W + new_col * W + new_row;
      f_out[j] = f_in[i];
    }
  }
}


/*Central element of the old_filter in the (0,0,0) position of the new_filter.
 *(x,y,z) -> ((x-X/2)%X, (y-Y/2)%Y, (z-Z/2)%Z)
 *new_filter[RHS] = old_filter[LHS]
 * <<H,W,D>>
 */
__global__ void align_filer(float* f_in, float* f_out, int H, int W, int D, int out_size)
{
  int col = blockIdx.x*blockDim.x+threadIdx.x;
  int row = blockIdx.y*blockDim.y+threadIdx.y;
  int dep = blockIdx.z*blockDim.z+threadIdx.z;

  int new_col = ((col - H/2) % H);
  int new_row = ((row - W/2) % W);
  int new_dep = ((dep - D/2) % D);

  new_col = new_col < 0 ? H + new_col: new_col;
  new_row = new_row < 0 ? W + new_row: new_row;
  new_dep = new_dep < 0 ? D + new_dep: new_dep;

  if(col < H && row < W && dep < D) {
    #pragma unroll
    for(int itr = 0; itr < out_size; itr++)  
    {
      int i = itr * D * H * W + dep * H * W + col * W + row;
      int j = itr * D * H * W + new_dep * H * W + new_col * W + new_row;
      f_out[j] = f_in[i];
    }
  }
}


/*This utility function can be used to pad the filter(as it needs to be of the size of the input)
 * or to pad the input when required
 *<H+2*pad, W+2*pad,D>
 */

__global__ void pad_input_(float* f_in, float* f_out, int H, int W, int D, int pad_front, int pad_back, int BS) {
  int col = blockIdx.x*blockDim.x+threadIdx.x;
  int row = blockIdx.y*blockDim.y+threadIdx.y;
  int dep = blockIdx.z*blockDim.z+threadIdx.z;

  int new_H = H+ pad_front + pad_back; int new_W = W + pad_front + pad_back; 

  if(col < new_H && row < new_W && dep < D) 
  {
    #pragma unroll
    for(int itr = 0; itr < BS; itr++)
    {
        int i = itr * D * new_H * new_W + dep * new_H * new_W + col * new_W + row;
        int j = itr * D * H * W + dep * H * W + (col - pad_front) *W+ (row - pad_front) ;

        if((col < pad_front || col > H+pad_front-1) || (row < pad_front || row > W+pad_front-1)) f_out[i] = 0;
        else f_out[i] = f_in[j];  
    }
  }
}

/* Croping and striding the required output from the complete (padded) output
 * <H,W,1>, <256>
 */
__global__ void crop_and_stride(float* f_in, float* f_out, int H, int W, int nos_oH, int nos_oW, int D, int stride, int out_size) {
  int col = blockIdx.x*blockDim.x+threadIdx.x;
  int row = blockIdx.y*blockDim.y+threadIdx.y;
  int batch = blockIdx.z*blockDim.z+threadIdx.z;
  int i =  batch*D*H*W + ((D-1)/2) * H * W + col * W + row;
  
  int crop_H_b = (H - nos_oH)/2; int crop_H_f;
  int crop_W_b = (W - nos_oW)/2; int crop_W_f;
  if((H - nos_oH) % 2 == 0) crop_H_f = crop_H_b; else crop_H_f = crop_H_b + 1;
  if((W - nos_oW) % 2 == 0) crop_W_f = crop_W_b; else crop_W_f = crop_W_b + 1;

  int j = batch * nos_oH * nos_oW + (col - crop_H_f) * nos_oW+ (row - crop_W_f);

  if(col < H && row < W && batch < out_size) {
      if(col >= crop_H_f && col < H - crop_H_b && row >= crop_W_f && row < W - crop_W_b)
      {
        if(stride == 1)f_out[j] = f_in[i];
        else 
        {    
          if(((col - crop_H_f)%stride) == 0 && ((row - crop_W_f)%stride == 0))
          {
              j = batch * (nos_oH/stride + 1) * (nos_oW/stride + 1) + (((col - crop_H_f)/stride) * (nos_oW/stride + 1)) + ((row - crop_W_f)/stride);
              f_out[j] = f_in[i];
          }
        }
      }
  }
}

__global__ void replicate_input(cufftComplex* input_layer_pad, cufftComplex* d_outA, int size, int H, int W, int D) {
  int i = blockIdx.x *blockDim.x + threadIdx.x;
  if(i < size)
  {
    d_outA[i] = input_layer_pad[i % (D * (H/2 + 1) * W)];  
  }
}

/* input arguments are input_layer, padding, stride, batch_size, input_layer dimensions
 * Operations: pad(input), convert_to_frequency_domain(padded_input_layer)
 * Output is the the input layer after padding and converted to the frequency domain by FFT
 */
float* conv_operation(cufftComplex* filter_align, cufftComplex* input_layer_pad, int H, int W, int D, int OS, float& conv_time, float& overhead_time) {  
  float milliseconds = 0;
  
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  int new_OS = OS+ 1;
  int N[3] = {D,H,W};
  cufftReal* d_inA;
  cufftComplex *d_outA, *d_outB;
  cufftHandle bwplan;
  size_t real_size = new_OS * D* W * H * sizeof(cufftReal);
  size_t complex_size = new_OS * D * H * (W/2 + 1) * sizeof(cufftComplex);

  cudaMalloc((void**)&d_inA, real_size);
  
  cudaMalloc((void**)&d_outA, complex_size);
  
  cudaMemset(d_inA,0,real_size);

  // cudaEventRecord(start);
  // for(int  i = 0; i < new_OS; i++) {
  //     cudaMemcpyAsync(&d_outA[i * D * (H/2 + 1) * W], input_layer_pad,   D * H * (W/2 + 1) * sizeof(cufftComplex),  cudaMemcpyDeviceToDevice);
  // }
  // cudaEventRecord(stop);
  // cudaEventSynchronize(stop);
  // milliseconds = 0;
  // cudaEventElapsedTime(&milliseconds, start, stop);
  // overhead_time += milliseconds;

  int blocksxc = ceil((new_OS * D * (H/2 + 1) * W) / 1024.0f);
  dim3 threads4c(1024);
  dim3 grid4c(blocksxc);
  
  cudaEventRecord(start);
  replicate_input<<<grid4c, threads4c>>>(input_layer_pad, d_outA, (new_OS * D * (H/2 + 1) * W), H, W, D);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  overhead_time += milliseconds;
  
  /* Make the plans for filter and inverse of output */
  cudaEventRecord(start);
  cufftSafeCall(cufftPlanMany(&bwplan, 3, N, NULL, 0,0,NULL,0,0, CUFFT_C2R ,new_OS));
  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  conv_time += milliseconds;
  
  /* FFT on the filter */
  d_outB = filter_align;

  int blocksx = ceil((new_OS * D*H*(W/2 + 1)) / 1024.0f);
  dim3 threads4(1024);
  dim3 grid4(blocksx);
  
  cudaEventRecord(start);
  pointwise_product<<<grid4, threads4>>>(d_outA, d_outB, (new_OS * D*H*(W/2 + 1)), 1.0f/(H*W*D));
  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  conv_time += milliseconds;

  /* Inverse FFT of output */
  cudaEventRecord(start);
  cufftSafeCall(cufftExecC2R(bwplan, d_outA, d_inA));
  cudaEventRecord(stop);
  
  cudaEventSynchronize(stop);
  milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  conv_time += milliseconds;
  

  /* Free the GPU */  
  cudaFree(d_outA); 

  cufftDestroy(bwplan);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  
  return d_inA;
}

/* input arguments are input_layer, padding, stride, batch_size, input_layer dimensions
 * Operations: pad(input), convert_to_frequency_domain(padded_input_layer)
 * Output is the the input layer after padding and converted to the frequency domain by FFT
 */
cufftComplex* pre_processinput(float* input_layer, int pad, int  batch_size, int* il_dim, float& conv_time, float& overhead_time) {
  cudaError_t err = cudaSuccess;
  int H = il_dim[0], W = il_dim[1], D = il_dim[2]; int BS = batch_size;

  float milliseconds = 0;
  
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  
  /* pad input */
  int new_H = H+2*pad; int new_W = W+2*pad;
  float *pad_input_in = NULL; cudaMalloc((void **)&pad_input_in, BS * H * W * D * sizeof(float));
  float *pad_input_out = NULL; cudaMalloc((void **)&pad_input_out, BS * new_H * new_W * D * sizeof(float));  

  
  cudaMemcpy(pad_input_in, input_layer, BS * H * W * D * sizeof(float) , cudaMemcpyHostToDevice);
  
  dim3 threads1(8,8,8);
  dim3 grid1(ceil(new_H/8.0f),ceil(new_W/8.0f),ceil(D/8.0f));

  cudaEventRecord(start);
  pad_input_<<<grid1,threads1>>>(pad_input_in, pad_input_out, H,W,D,pad, pad, BS);
  cudaEventRecord(stop);    
  
  err = cudaGetLastError(); if(err!=cudaSuccess){fprintf(stderr, "Failed to launch pad input (error code %s)!\n", cudaGetErrorString(err)); exit(EXIT_FAILURE);}

  cudaEventSynchronize(stop);
  milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  overhead_time += milliseconds;
  
  cudaFree(pad_input_in); 
  
  H = new_H; W = new_W;
  int N[3] = {D,H, W};
  
  cufftComplex* d_input_complex;
  cufftHandle fwplan_input;
  // size_t real_size = BS * D* W * H * sizeof(cufftReal);
  size_t complex_size = BS * D * W * (H/2 + 1) * sizeof(cufftComplex);

  cudaMalloc((void**)&d_input_complex, complex_size);
  
  cudaEventRecord(start);
  cufftSafeCall(cufftPlanMany(&fwplan_input, 3, N, NULL, 0,0,NULL,0,0, CUFFT_R2C ,BS));
  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  conv_time += milliseconds;


  cudaEventRecord(start);
  cufftSafeCall(cufftExecR2C(fwplan_input, pad_input_out, d_input_complex));
  cudaEventRecord(stop);
  

  cudaEventSynchronize(stop);
  milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  conv_time += milliseconds;

  cudaFree(pad_input_out);
  cufftDestroy(fwplan_input);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return d_input_complex;
}


cufftComplex* pre_process_filter(float* kernel, int pad, int* il_dim, int* kernel_dim, int out_size, float& conv_time, float& overhead_time)
{
    int H = il_dim[0], W = il_dim[1], D = il_dim[2];
    int fH = kernel_dim[0], fW = kernel_dim[1] , fD = kernel_dim[2];
    cudaError_t err = cudaSuccess;
    int new_H = H+2*pad; int new_W = W+2*pad;
    H = new_H; W = new_W;

    float milliseconds = 0;
  
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    /*flip filter */
    float *f_A = NULL; cudaMalloc((void **)&f_A, out_size * fH * fW * fD * sizeof(float));
    float *f_B = NULL; cudaMalloc((void **)&f_B, out_size * fH * fW * fD * sizeof(float));
    cudaMemcpy(f_A, kernel , out_size * fH * fW * fD * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threads0(8,8,8);
    dim3 grid0(ceil(fH/8.0f),ceil(fW/8.0f),ceil(fD/8.0f));

    cudaEventRecord(start);
    flip_filer<<<grid0, threads0>>>(f_A, f_B, fH,fW,fD, out_size);
    cudaEventRecord(stop);

    err = cudaGetLastError(); if(err!=cudaSuccess){fprintf(stderr, "Failed to launch align_filter(error code %s)!\n", cudaGetErrorString(err)); exit(EXIT_FAILURE);}
    cudaFree(f_A);

    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    overhead_time += milliseconds;

   /* flip filter end */

    /* pad filter */
    int bpad = (new_H - fH)/2;
    int fpad; if((new_H - fH) % 2 == 0) fpad = bpad; else fpad = bpad + 1;  
    int new_fH = fH+fpad+bpad; int new_fW = fW+fpad+bpad;
    float *pad_filter_in = NULL;
    float *pad_filter_out = NULL; cudaMalloc((void **)&pad_filter_out, out_size * new_fH * new_fW * D * sizeof(float));

    pad_filter_in = f_B;
    dim3 threads2(8,8,8);
    dim3 grid2(ceil(new_fH/8.0f),ceil(new_fW/8.0f),ceil(D/8.0f));

    cudaEventRecord(start);
    pad_input_<<<grid2,threads2>>>(pad_filter_in, pad_filter_out, fH,fW,D,fpad,bpad, out_size);
    cudaEventRecord(stop);

    err = cudaGetLastError(); if(err!=cudaSuccess){fprintf(stderr, "Failed to launch pad filter(error code %s)!\n", cudaGetErrorString(err)); exit(EXIT_FAILURE);}

    fH = new_fH; fW = new_fW;
    cudaFree(pad_filter_in);
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    overhead_time += milliseconds;
    
    /* pad filter end */
    float *d_A = NULL;
    float *d_B = NULL; cudaMalloc((void **)&d_B, out_size * fH * fW * fD * sizeof(float));
    d_A = pad_filter_out;

    dim3 threads3(8,8,8);
    dim3 grid3(ceil(fH/8.0f),ceil(fW/8.0f),ceil(fD/8.0f));

    cudaEventRecord(start);
    align_filer<<<grid3, threads3>>>(d_A, d_B, fH,fW,fD,out_size);
    cudaEventRecord(stop);

    err = cudaGetLastError(); if(err!=cudaSuccess){fprintf(stderr, "Failed to launch align_filter(error code %s)!\n", cudaGetErrorString(err)); exit(EXIT_FAILURE);}
    cudaFree(d_A); 

    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    overhead_time += milliseconds;
    /* align filter end */

    int N[3] = {D ,H, W};
    cufftComplex* d_input_complex;
    cufftHandle fwplan_input;
    int new_OS = out_size + 1;
    // size_t real_size = out_size * D* W * H * sizeof(cufftReal);
    size_t complex_size = new_OS * D * W * (H/2 + 1) * sizeof(cufftComplex);

    cudaMalloc((void**)&d_input_complex, complex_size);
    cudaMemset(d_input_complex, 0, complex_size);

    cudaEventRecord(start);
    cufftSafeCall(cufftPlanMany(&fwplan_input, 3, N, NULL, 0,0,NULL,0,0, CUFFT_R2C ,out_size));
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    conv_time += milliseconds;

    cudaEventRecord(start);
    cufftSafeCall(cufftExecR2C(fwplan_input, d_B, d_input_complex));
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    conv_time += milliseconds;

    cudaFree(d_B);
    cufftDestroy(fwplan_input); 

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    

    return d_input_complex;

}

/* input arguments are input_layer, kernel, padding, stride, batch_size, input_layer dimensions, kernel dimensions
 * Operations: pad(filter), align(filter), output = convolve(input,filter), crop(output), stride(output)
 * Output is the convolvolution of that filter and the whole input
 */
float* convolve_FFT(cufftComplex* input_layer_pad, cufftComplex* kernel, int pad, int stride, int batch_size, int* il_dim, int* kernel_dim, int out_size,
                                                                              float& conv_time, float& overhead_time) {
  /* initializations */
  int H = il_dim[0], W = il_dim[1], D = il_dim[2]; 
  int fH = kernel_dim[0], fW = kernel_dim[1];
  cudaError_t err = cudaSuccess;
    
  int new_H = H+2*pad; int new_W = W+2*pad;
  H = new_H; W = new_W;
  int bpad = (new_H - fH)/2;
  int fpad; if((new_H - fH) % 2 == 0) fpad = bpad; else fpad = bpad + 1;  
  int new_fH = fH+fpad+bpad; int new_fW = fW+fpad+bpad;
  fH = new_fH; fW = new_fW;

  float milliseconds = 0;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  /* Convolve begin (FFT, Pointwise prodcut, IFFT) */
  float* conv_result = conv_operation(kernel, input_layer_pad, H, W, D, out_size, conv_time, overhead_time);
  /* convolve end */

  /* crop output */
  fH = kernel_dim[0]; fW = kernel_dim[1] ;
  int oH = (H - fH)/stride + 1; int oW = (W - fW)/stride + 1;
  int nos_oH = (H - fH + 1); int nos_oW = W -fW + 1;
  float* result2 = (float*)malloc((out_size) * oW*oH* sizeof(float));
  float *crop_out = NULL; err = cudaMalloc((void **)&crop_out, out_size * oH * oW * sizeof(float));
  if(err!=cudaSuccess){fprintf(stderr, "Failed to allocate memory crop_out (error code %s)!\n", cudaGetErrorString(err)); exit(EXIT_FAILURE);}

  float *crop_in = NULL; crop_in = conv_result; 

  dim3 threads5(8,8,8);
  dim3 grid5(ceil(H/8.0f),ceil(W/8.0f),ceil(out_size/8.0f));

  cudaEventRecord(start);
  crop_and_stride<<<grid5, threads5>>>(crop_in, crop_out, H, W, nos_oH, nos_oW, D, stride, out_size);
  cudaEventRecord(stop);

  err = cudaGetLastError(); if(err!=cudaSuccess){fprintf(stderr, "Failed to launch crop(error code %s)!\n", cudaGetErrorString(err)); exit(EXIT_FAILURE);}
  
  cudaMemcpy(result2, crop_out, out_size* oW*oH* sizeof(float) ,cudaMemcpyDeviceToHost);
  cudaEventSynchronize(stop);
  milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  overhead_time += milliseconds;
  
  cudaFree(crop_in); cudaFree(crop_out);
  /* crop output end */

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return result2;
}

/* Implementation of the forward pass of FFT Kernel */
float* FFT::forward(int out_size, int channel, int kernel_height, int kernel_width, int pad, int stride, float* kernel, 
                    int batch_size, int height, int width, float* input_layer_without_padding, float& conv_time, float& overhead_time) {
  int il_dim[3] = {height, width, channel}; int kernel_dim[3] = {kernel_height, kernel_width, channel};
  int out_H = ((height - kernel_height + 2 * pad)/stride) + 1; 
  int out_W = ((width - kernel_width + 2 * pad)/stride) + 1; 

  conv_time = 0;
  overhead_time = 0;
  float milliseconds = 0;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  int H = height+2*pad; int W = width+2*pad;

  cufftComplex* input_layer_pad = pre_processinput(input_layer_without_padding, pad, batch_size, il_dim, conv_time, overhead_time);
  cufftComplex* filter_complex = pre_process_filter(kernel, pad, il_dim, kernel_dim, out_size, conv_time, overhead_time);

  float* final_output = (float *)malloc(batch_size * out_size * out_H * out_W * sizeof(float)); 
  
  for(int l = 0; l < batch_size ; l++) {
    float* actual_result = convolve_FFT(&input_layer_pad[l * channel * (H/2 + 1) * W], filter_complex, pad, stride, batch_size , il_dim, kernel_dim,
                                        out_size, conv_time, overhead_time);
    
    cudaEventRecord(start);
    #pragma unroll
    for(int ll = 0; ll < out_size; ll++) {
      for(int ii = 0; ii < out_H; ii++) {
        for(int jj = 0; jj < out_W; jj++) {
          final_output[l*out_size*out_H*out_W + ll*out_H*out_W + ii * out_W + jj] = actual_result[ll*out_H * out_W + ii*out_W + jj];
        }
      }
    }
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    overhead_time += milliseconds;
    
    free(actual_result);
  }
  cudaFree(input_layer_pad);
  cudaFree(filter_complex);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return final_output;
}