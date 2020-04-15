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
 __global__ void flip_filer(float* f_in, float* f_out, int H, int W, int D)
{
  int col = blockIdx.x*blockDim.x+threadIdx.x;
  int row = blockIdx.y*blockDim.y+threadIdx.y;
  int dep = blockIdx.z*blockDim.z+threadIdx.z;
  int i = dep * H * W + col * W + row ;

  int new_col = H - col -1;
  int new_row = W - row - 1;
  int new_dep = D - dep - 1;

  int j = new_dep * H * W + new_col * W + new_row;

  if(col < H && row < W && dep < D) {
      f_out[j] = f_in[i];
  }
}


/*Central element of the old_filter in the (0,0,0) position of the new_filter.
 *(x,y,z) -> ((x-X/2)%X, (y-Y/2)%Y, (z-Z/2)%Z)
 *new_filter[RHS] = old_filter[LHS]
 * <<H,W,D>>
 */
__global__ void align_filer(float* f_in, float* f_out, int H, int W, int D)
{
  int col = blockIdx.x*blockDim.x+threadIdx.x;
  int row = blockIdx.y*blockDim.y+threadIdx.y;
  int dep = blockIdx.z*blockDim.z+threadIdx.z;
  int i = dep * H * W + col * W + row;

  int new_col = ((col - H/2) % H);
  int new_row = ((row - W/2) % W);
  int new_dep = ((dep - D/2) % D);

  new_col = new_col < 0 ? H + new_col: new_col;
  new_row = new_row < 0 ? W + new_row: new_row;
  new_dep = new_dep < 0 ? D + new_dep: new_dep;

  int j = new_dep * H * W + new_col * W + new_row;

  if(col < H && row < W && dep < D) {
    f_out[j] = f_in[i];
  }
}


/*This utility function can be used to pad the filter(as it needs to be of the size of the input)
 * or to pad the input when required
 *<H+2*pad, W+2*pad,D>
 */
__global__ void pad_input(float* f_in, float* f_out, int H, int W, int D, int pad_front, int pad_back) {
  int col = blockIdx.x*blockDim.x+threadIdx.x;
  int row = blockIdx.y*blockDim.y+threadIdx.y;
  int dep = blockIdx.z*blockDim.z+threadIdx.z;

  int new_H = H+ pad_front + pad_back; int new_W = W + pad_front + pad_back; 

  int i = dep * new_H * new_W + col * new_W + row;
  int j = dep * H * W + (col - pad_front) *W+ (row - pad_front) ;

  if(col < new_H && row < new_W && dep < D) {
    if((col < pad_front || col > H+pad_front-1) || (row < pad_front || row > W+pad_front-1)) f_out[i] = 0;
    else f_out[i] = f_in[j];
  }
}

/* Croping the required output from the complete (padded) output
 * <H,W,1>, <256>
 */
__global__ void crop(float* f_in, float* f_out, int H, int W, int O_H, int O_W, int D) {
  int col = blockIdx.x*blockDim.x+threadIdx.x;
  int row = blockIdx.y*blockDim.y+threadIdx.y;
  int i =  ((D-1)/2) * H * W + col * W + row;
  
  int crop_H_b = (H - O_H)/2; int crop_H_f;
  int crop_W_b = (W - O_W)/2; int crop_W_f;
  if((H - O_H) % 2 == 0) crop_H_f = crop_H_b; else crop_H_f = crop_H_b + 1;
  if((W - O_W) % 2 == 0) crop_W_f = crop_W_b; else crop_W_f = crop_W_b + 1;

  int j = (col - crop_H_f) * O_W+ (row - crop_W_f);

  if(col < H && row < W) {
      if(col >= crop_H_f && col < H - crop_H_b && row >= crop_W_f && row < W - crop_W_b)f_out[j] = f_in[i];
  }
}


/*stride and output the required size output (O = ((W - k + 2 * P)/stride) + 1)
 *call this only if stride is not 1
 *<<H, W, 1>>
 */
__global__ void stride_(float* f_in, float* f_out, int H, int W, int stride) {
  int col = blockIdx.x*blockDim.x+threadIdx.x;
  int row = blockIdx.y*blockDim.y+threadIdx.y;

  int O_W = W/stride + 1;
  
  int i = col * W + row; 

  if(col < H && row < W && (col%stride == 0) && (row%stride == 0)) {
      int j = (col/stride) * O_W + (row/stride) ;
      f_out[j] = f_in[i];
  }
}

/* input arguments are input_layer, padding, stride, batch_size, input_layer dimensions
 * Operations: pad(input), convert_to_frequency_domain(padded_input_layer)
 * Output is the the input layer after padding and converted to the frequency domain by FFT
 */
float* conv_operation(float* filter_align, float* input_layer_pad, int H, int W, int D, int BS, float& conv_time, float& overhead_time) {

  float milliseconds = 0;
  
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  int new_BS = BS + 1;
  int N[3] = {D,H,W};
  cufftReal* d_inA;
  cufftComplex *d_outA, *d_outB;
  cufftHandle fwplanA, bwplan;
  size_t real_size = new_BS * D* W * H * sizeof(cufftReal);
  size_t complex_size = new_BS * D * H * (W/2 + 1) * sizeof(cufftComplex);

  cudaMalloc((void**)&d_inA, real_size);
  
  cudaMalloc((void**)&d_outA, complex_size);
  cudaMalloc((void**)&d_outB, complex_size);
  
  cudaMemset(d_inA,0,real_size);
  cudaMemset(d_outB,0,complex_size);

  float * filter_align_in = (float *)malloc(real_size);
  for(int  i = 0; i < new_BS; i++) {
      cudaMemcpy(&filter_align_in[i * D * H * W], filter_align,  D* W * H * sizeof(cufftReal),  cudaMemcpyHostToHost);
  }

  cudaMemcpy(d_inA, filter_align_in, real_size,  cudaMemcpyHostToDevice);
  
  /* Make the plans for filter and inverse of output */
  cudaEventRecord(start);
  cufftSafeCall(cufftPlanMany(&fwplanA, 3, N, NULL, 0,0,NULL,0,0, CUFFT_R2C ,new_BS));
  cufftSafeCall(cufftPlanMany(&bwplan, 3, N, NULL, 0,0,NULL,0,0, CUFFT_C2R ,new_BS));
  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  conv_time += milliseconds;
  
  /* FFT on the filter */
  cudaEventRecord(start);
  cufftSafeCall(cufftExecR2C(fwplanA, d_inA, d_outA));
  cudaEventRecord(stop);

  cudaMemcpy(d_outB, input_layer_pad, (BS) * D * H * (W/2 + 1) * sizeof(cufftComplex), cudaMemcpyHostToDevice); 

  cudaEventSynchronize(stop);
  milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  conv_time += milliseconds;

  int blocksx = ceil((new_BS * D*H*(W/2 + 1)) / 256.0f);
  dim3 threads4(256);
  dim3 grid4(blocksx);
  
  /* PointWise Product */
  cudaEventRecord(start);
  pointwise_product<<<grid4, threads4>>>(d_outA, d_outB, (new_BS * D*H*(W/2 + 1)), 1.0f/(H*W*D));
  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  conv_time += milliseconds;

  /* Inverse FFT of output */
  cudaEventRecord(start);
  cufftSafeCall(cufftExecC2R(bwplan, d_outA, d_inA));
  cudaEventRecord(stop);

  float* result1 = (float*)malloc((BS) * D * H * W * sizeof(float));
  cudaMemcpy(result1, d_inA, (BS) * D * H * W * sizeof(float), cudaMemcpyDeviceToHost);
  
  cudaEventSynchronize(stop);
  milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  conv_time += milliseconds;


  /* Free the GPU */
  cudaFree(d_inA); 
  
  cudaFree(d_outA); 
  cudaFree(d_outB);
  
  free(filter_align_in);
  
  cufftDestroy(fwplanA);
  cufftDestroy(bwplan);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  
  return result1;
}

/* input arguments are input_layer, padding, stride, batch_size, input_layer dimensions
 * Operations: pad(input), convert_to_frequency_domain(padded_input_layer)
 * Output is the the input layer after padding and converted to the frequency domain by FFT
 */
float* pre_processinput(float* input_layer, int pad, int  batch_size, int* il_dim, float& conv_time, float& overhead_time) {
  cudaError_t err = cudaSuccess;
  int H = il_dim[0], W = il_dim[1], D = il_dim[2]; int BS = batch_size;
  
  float milliseconds = 0;
  
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  /* pad input */
  int new_H = H+2*pad; int new_W = W+2*pad;
  float *input_layer_pad = (float *)malloc(BS * D * new_H* new_W *sizeof(float));
  float *pad_input_in = NULL; cudaMalloc((void **)&pad_input_in, H * W * D * sizeof(float));
  float *pad_input_out = NULL; cudaMalloc((void **)&pad_input_out, new_H * new_W * D * sizeof(float));  

  for(int i = 0; i < BS; i++) {
    cudaMemcpy(pad_input_in, &input_layer[i * D * H * W], H * W * D * sizeof(float) , cudaMemcpyHostToDevice);
    
    dim3 threads1(1,1,1);
    dim3 grid1(new_H,new_W,D);

    cudaEventRecord(start);
    pad_input<<<grid1,threads1>>>(pad_input_in, pad_input_out, H,W,D,pad, pad);
    cudaEventRecord(stop);
    
    err = cudaGetLastError(); if(err!=cudaSuccess){fprintf(stderr, "Failed to launch pad input (error code %s)!\n", cudaGetErrorString(err)); exit(EXIT_FAILURE);}
    
    cudaMemcpy(&input_layer_pad[i * D * new_H * new_W], pad_input_out , new_H * new_W * D * sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    overhead_time += milliseconds;  
  }
  cudaFree(pad_input_in); cudaFree(pad_input_out);


  H = new_H; W = new_W;
  int N[3] = {D,H, W};
  
  cufftReal* d_input;
  cufftComplex* d_input_complex;
  cufftHandle fwplan_input;
  size_t real_size = BS * D* W * H * sizeof(cufftReal);
  size_t complex_size = BS * D * W * (H/2 + 1) * sizeof(cufftComplex);

  float* complex_input = (float*)malloc(complex_size);

  cudaMalloc((void**)&d_input, real_size);
  cudaMalloc((void**)&d_input_complex, complex_size);
  cudaMemset(d_input,0,real_size);

  cudaMemcpy(d_input, input_layer_pad, real_size, cudaMemcpyHostToDevice);
  
  cudaEventRecord(start);
  cufftSafeCall(cufftPlanMany(&fwplan_input, 3, N, NULL, 0,0,NULL,0,0, CUFFT_R2C ,BS));
  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  conv_time += milliseconds;


  cudaEventRecord(start);
  cufftSafeCall(cufftExecR2C(fwplan_input, d_input, d_input_complex));
  cudaEventRecord(stop);
  
  cudaMemcpy(complex_input, d_input_complex , complex_size, cudaMemcpyDeviceToHost);

  cudaEventSynchronize(stop);
  milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  conv_time += milliseconds;

  cudaFree(d_input);
  cufftDestroy(fwplan_input);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return complex_input;
}

/* input arguments are input_layer, kernel, padding, stride, batch_size, input_layer dimensions, kernel dimensions
 * Operations: pad(filter), align(filter), output = convolve(input,filter), crop(output), stride(output)
 * Output is the convolvolution of that filter and the whole input
 */
float* convolve_FFT(float* input_layer_pad, float * kernel, int pad, int stride, int batch_size, int* il_dim, int* kernel_dim, 
                    float& conv_time, float& overhead_time) {
  /* initializations */
  int H = il_dim[0], W = il_dim[1], D = il_dim[2]; int BS = batch_size;
  int fH = kernel_dim[0], fW = kernel_dim[1] , fD = kernel_dim[2];
  cudaError_t err = cudaSuccess;

  float milliseconds = 0;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  
  int new_H = H+2*pad; int new_W = W+2*pad;
  H = new_H; W = new_W;

  /* flip filter */
  float *f_A = NULL; cudaMalloc((void **)&f_A, fH * fW * fD * sizeof(float));
  float *f_B = NULL; cudaMalloc((void **)&f_B, fH * fW * fD * sizeof(float));
  cudaMemcpy(f_A, kernel , fH * fW * fD * sizeof(float), cudaMemcpyHostToDevice);

  dim3 threads0(1,1,1);
  dim3 grid0(fH,fW,fD);

  cudaEventRecord(start);
  flip_filer<<<grid0, threads0>>>(f_A, f_B, fH,fW,fD);
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
  float *pad_filter_out = NULL; cudaMalloc((void **)&pad_filter_out, new_fH * new_fW * D * sizeof(float));

  pad_filter_in = f_B;
  dim3 threads2(1,1,1);
  dim3 grid2(new_fH,new_fW,D);

  cudaEventRecord(start);
  pad_input<<<grid2,threads2>>>(pad_filter_in, pad_filter_out, fH,fW,D,fpad,bpad);
  cudaEventRecord(stop);

  err = cudaGetLastError(); if(err!=cudaSuccess){fprintf(stderr, "Failed to launch pad filter(error code %s)!\n", cudaGetErrorString(err)); exit(EXIT_FAILURE);}

  fH = new_fH; fW = new_fW;
  cudaFree(pad_filter_in);

  cudaEventSynchronize(stop);
  milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  overhead_time += milliseconds;
  /* pad filter end */
 
  /* align filter begin */
  float *filter_align = (float *)malloc(fH * fW * fD *sizeof(float));
  float *d_A = NULL;
  float *d_B = NULL; cudaMalloc((void **)&d_B, fH * fW * fD * sizeof(float));
  d_A = pad_filter_out;

  dim3 threads3(1,1,1);
  dim3 grid3(fH,fW,fD);

  cudaEventRecord(start);
  align_filer<<<grid3, threads3>>>(d_A, d_B, fH,fW,fD);
  cudaEventRecord(stop);

  err = cudaGetLastError(); if(err!=cudaSuccess){fprintf(stderr, "Failed to launch align_filter(error code %s)!\n", cudaGetErrorString(err)); exit(EXIT_FAILURE);}
  
  cudaMemcpy(filter_align, d_B, fH*fW*fD*sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(d_A); cudaFree(d_B);
  
  cudaEventSynchronize(stop);
  milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  overhead_time += milliseconds;

  /* align filter end */
 
  /* Convolve begin (FFT, Pointwise prodcut, IFFT) */
  float* conv_result = conv_operation( filter_align, input_layer_pad, H, W, D, BS, conv_time, overhead_time);
  free(filter_align);
  /* convolve end */

  /* crop output */
  fH = kernel_dim[0]; fW = kernel_dim[1] ; fD = kernel_dim[2];
  int oH = H - fH + 1; int oW = W - fW + 1;
  float* result2 = (float*)malloc((BS) * oW*oH* sizeof(float));
  float *crop_out = NULL; 
  err = cudaMalloc((void **)&crop_out, oH * oW * sizeof(float));
  if(err!=cudaSuccess){fprintf(stderr, "Failed to allocate memory crop_out (error code %s)!\n", cudaGetErrorString(err)); exit(EXIT_FAILURE);}

  float *crop_in = NULL; 
  err = cudaMalloc((void **)&crop_in, D * H * W * sizeof(float));  
  if(err!=cudaSuccess){fprintf(stderr, "Failed to allocate memory crop_in (error code %s)!\n", cudaGetErrorString(err)); exit(EXIT_FAILURE);}

  for(int i = 0; i < BS; i++) {
    cudaMemcpy(crop_in, &conv_result[i * D * H * W],  D * H * W * sizeof(float),cudaMemcpyHostToDevice);
    err = cudaGetLastError(); if(err!=cudaSuccess){fprintf(stderr, "Failed to copy(error code %s)!\n", cudaGetErrorString(err)); exit(EXIT_FAILURE);}

    dim3 threads5(1,1,1);
    dim3 grid5(H,W,1);

    cudaEventRecord(start);
    crop<<<grid5, threads5>>>(crop_in, crop_out, H, W, oH, oW, D);
    cudaEventRecord(stop);

    err = cudaGetLastError(); if(err!=cudaSuccess){fprintf(stderr, "Failed to launch crop(error code %s)!\n", cudaGetErrorString(err)); exit(EXIT_FAILURE);}
    
    cudaMemcpy(&result2[i*oW*oH], crop_out, oW*oH* sizeof(float) ,cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    conv_time += milliseconds;
  }
  cudaFree(crop_in); cudaFree(crop_out);
  free(conv_result);
  /* crop output end */

  /* stride output stride_(float* f_in, float* f_out, int H, int W, int stride) */
  if(stride != 1) {
    int sH = oH / stride + 1; int sW = oW / stride + 1; 
    float* result_s = (float *)malloc(BS* sH*sW*sizeof(float));

    for(int i = 0; i < BS ; i++) {
      float *stride_in = NULL; cudaMalloc((void **)&stride_in, oH * oW * sizeof(float));
      float *stride_out = NULL; cudaMalloc((void **)&stride_out, sH * sW * sizeof(float));  
      cudaMemcpy(stride_in, &result2[i * oW* oH], oW*oH* sizeof(float) ,cudaMemcpyHostToDevice);
      dim3 threads6(1,1,1);
      dim3 grid6(oH,oW,1);

      cudaEventRecord(start);
      stride_<<<grid6, threads6>>>(stride_in, stride_out ,oH, oW, stride);
      cudaEventRecord(stop);

      err = cudaGetLastError(); if(err!=cudaSuccess){fprintf(stderr, "Failed to launch stride(error code %s)!\n", cudaGetErrorString(err)); exit(EXIT_FAILURE);}
      cudaMemcpy(&result_s[i*sH*sW], stride_out , sH * sW * sizeof(float) ,cudaMemcpyDeviceToHost);
      cudaFree(stride_in); cudaFree(stride_out);

      cudaEventSynchronize(stop);
      milliseconds = 0;
      cudaEventElapsedTime(&milliseconds, start, stop);
      overhead_time += milliseconds;
    }
    free(result2);
    result2 = result_s;
  }
  /* stride output end */

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  
  return result2;
}

/* Implementation of the forward pass of FFT Kernel */
float* FFT::forward(int out_size, int channel, int kernel_height, int kernel_width, int pad, int stride, float* kernel, 
                    int batch_size, int height, int width, float* input_layer_without_padding, float& conv_time, float& overhead_time) {
  int il_dim[3] = {height, width, channel}; int kernel_dim[3] = {kernel_height, kernel_width, channel};

  conv_time = 0;
  overhead_time = 0;
  float milliseconds = 0;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  int out_H = ((height - kernel_height + 2 * pad)/stride) + 1; 
  int out_W = ((width - kernel_width + 2 * pad)/stride) + 1;     
  
  float* input_layer_pad_complex = pre_processinput(input_layer_without_padding, pad, batch_size, il_dim, conv_time, overhead_time);

  float* final_output = (float *)malloc(batch_size * out_size * out_H * out_W * sizeof(float)); 
  
  for(int l = 0; l < out_size ; l++) {
    float* actual_result = convolve_FFT(input_layer_pad_complex, &kernel[l * channel * kernel_height* kernel_width], pad, stride, batch_size , il_dim, kernel_dim,
                                        conv_time, overhead_time);
    cudaEventRecord(start);
    for(int ll = 0; ll < batch_size; ll++) {
      for(int ii = 0; ii < out_H; ii++) {
        for(int jj = 0; jj < out_W; jj++) {
          final_output[ll*out_size*out_H*out_W + l*out_H*out_W + ii * out_W + jj] = actual_result[ll*out_H * out_W + ii*out_W + jj];
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
  free(input_layer_pad_complex);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return final_output;
}