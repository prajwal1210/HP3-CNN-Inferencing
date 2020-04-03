#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>

//multiplication of 2 cufftComplex elements
//// One complex product for each thread scaled by total elements
__global__ void pointwise_product(cufftComplex* d_outA, cufftComplex* d_outB, float size, float scale)
{
    int i = blockIdx.x *blockDim.x + threadIdx.x;

    if (i < size)
    {
        float a,b;
        a = d_outA[i].x * d_outB[i].x - d_outA[i].y * d_outB[i].y;
        b = d_outA[i].x * d_outB[i].y + d_outA[i].y * d_outB[i].x;
        d_outA[i].x = a * scale;
        d_outA[i].y = b * scale;
    }
}

//Central element of the old_filter in the (0,0,0) position of the new_filter.
//(x,y,z) -> ((x-X/2)%X, (y-Y/2)%Y, (z-Z/2)%Z)
//new_filter[RHS] = old_filter[LHS]
// <<H,W,D>>
__global__ void align_filer(float* f_in, float* f_out, int H, int W, int D)
{
    int col = blockIdx.x*blockDim.x+threadIdx.x;
    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int dep = blockIdx.z*blockDim.z+threadIdx.z;
    int i = col + row * H + dep * H * W;
 
    int new_col = ((col - H/2) % H);
    int new_row = ((row - W/2) % W);
    int new_dep = ((dep - D/2) % D);
 
    new_col = new_col < 0 ? H + new_col: new_col;
    new_row = new_row < 0 ? W + new_row: new_row;
    new_dep = new_dep < 0 ? D + new_dep: new_dep;

    int j = new_col + new_row * H + new_dep * H * W;

    if(col < H && row < W && dep < D)
    {
        f_out[j] = f_in[i];
    }
}

//This utility function can be used to pad the filter(as it needs to be of the size of the input)
// or to pad the input when required
//<H+2*pad, W+2*pad,D>
__global__ void pad_input(float* f_in, float* f_out, int H, int W, int D, int pad)
{
    int col = blockIdx.x*blockDim.x+threadIdx.x;
    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int dep = blockIdx.z*blockDim.z+threadIdx.z;
    int i = col + row * H + dep * H * W;

    int new_H = H+2*pad; int new_W = W+2*pad; 

    if(col < new_H && row < new_W && dep < D)
    {
        if(col < pad || col > H+pad-1 || row < pad || row > W+pad-1) f_out[i] = 0;
        else f_out[i] = f_in[i];
    }
}

//crop and output the required size output (O = ((W - k + 2 * P)) + 1)
//<<H, W, D>>
__global__ void crop(float* f_in, float* f_out, int H, int W, int D, int O_H, int O_W)
{
    int col = blockIdx.x*blockDim.x+threadIdx.x;
    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int dep = blockIdx.z*blockDim.z+threadIdx.z;
    int i = col + row * H + dep * H * W;
    
    int crop_H = (H - O_H)/2;
    int crop_W = (W - O_W)/2;

    int j = (col - crop_H) + (row - crop_W) * (O_H);

    if(col < H && row < W && dep == D/2)
    {
        if(col >= crop_H && col < H - crop_H && row >= crop_W && row < W - crop_W)f_out[j] = f_in[i];
    }
}

//stride and output the required size output (O = ((W - k + 2 * P)/stride) + 1)
//call this only if stride is not 1
//<<H, W, 1>>
__global__ void stride(float* f_in, float* f_out, int H, int W, int stride)
{
    int col = blockIdx.x*blockDim.x+threadIdx.x;
    int row = blockIdx.y*blockDim.y+threadIdx.y;
    //int dep = blockIdx.z*blockDim.z+threadIdx.z;
    
    int i = col + row * H; 
    //+ dep * H * W;

    if(col < H && row < W && (col%stride == 0) && (row%stride == 0))
    {
        int j = col/stride + (row/stride) * H;
        // dep * H * W;
        f_out[j] = f_in[i];
    }
}



// input arguments are input_layer, kernel, padding, stride, batch_size, input_layer dimensions, kernel dimensions
// Operations: pad(input), pad(filter), align(filter), output = convolve(input,filter), crop(output), stride(output)  

float* convolve_FFT(float * input_layer, float * kernel, int pad, int stride, int batch_size, int* il_dim, int* kernel_dim)
{
  //////initializations
  int H = il_dim[0], W = il_dim[1], D = il_dim[2];
  int fH = kernel_dim[0], fW = kernel_dim[1] , fD = kernel_dim[2];
  cufftReal* d_inA, *d_inB;
  cufftComplex* d_outA, *d_outB;
  cufftHandle fwplanA, fwplanB, bwplan;
  cudaError_t err = cudaSuccess;
  //////////////////////////

  ///////pad input
  int new_H = H+2*pad; int new_W = W+2*pad;
  float *pad_input_in = NULL; cudaMalloc((void **)&pad_input_in, H * W * D * sizeof(float));
  float *pad_input_out = NULL; cudaMalloc((void **)&pad_input_out, new_H * new_W * D * sizeof(float));
  float *input_layer_pad = (float *)malloc(new_H* new_W * D *sizeof(float));

  cudaMemcpy(pad_input_in, input_layer, real_size, cudaMemcpyHostToDevice);
  dim3 threads(1,1,1);
  dim3 grid(new_H,new_W,D);
  pad_input<<<grid,threads>>>(pad_input_in, pad_input_out, H,W,D,pad);
  err = cudaGetLastError(); if(err!=cudaSuccess){fprintf(stderr, "Failed to launch pad input (error code %s)!\n", cudaGetErrorString(err)); exit(EXIT_FAILURE);}
  cudaMemcpy(input_layer_pad, pad_input_out , new_H * new_W * D * sizeof(float), cudaMemcpyDeviceToHost);
  H = new_H; W = new_W;
  //////pad input end

  ///////pad filter 
  int fpad = (new_H - fH)/2; 
  int new_fH = fH+2*fpad; int new_fW = fW+2*fpad;
  float *pad_filter_in = NULL; cudaMalloc((void **)&pad_filter_in, fH * fW * fD * sizeof(float));
  float *pad_filter_out = NULL; cudaMalloc((void **)&pad_filter_out, new_fH * new_fW * D * sizeof(float));
  float *filter_pad = (float *)malloc(new_fH* new_fW * D *sizeof(float));

  cudaMemcpy(pad_filter_in, kernel , real_size, cudaMemcpyHostToDevice);
  dim3 threads(1,1,1);
  dim3 grid(new_fH,new_fW,D);
  pad_input<<<grid,threads>>>(pad_filter_in, pad_filter_out, fH,fW,D,fpad);
  err = cudaGetLastError(); if(err!=cudaSuccess){fprintf(stderr, "Failed to launch pad filter(error code %s)!\n", cudaGetErrorString(err)); exit(EXIT_FAILURE);}
  cudaMemcpy(filter_pad, pad_filter_out , new_fH * new_fW * D * sizeof(float), cudaMemcpyDeviceToHost);
  fH = new_fH; fW = new_fW;
  //////pad input end

  ///////align filter begin
  float *filter_align = (float *)malloc(fH * fW * fD *sizeof(float));
  float *d_A = NULL; cudaMalloc((void **)&d_A, fH * fW * fD * sizeof(float));
  float *d_B = NULL; cudaMalloc((void **)&d_B, fH * fW * fD * sizeof(float));
  cudaMemcpy(d_A, filter_pad, fH * fW * fD * sizeof(float), cudaMemcpyHostToDevice);

  dim3 threads(1,1,1);
  dim3 grid(fH,fW,fD);
  align_filer<<<grid, threads>>>(d_A, d_B, fH,fW,fD);
  err = cudaGetLastError(); if(err!=cudaSuccess){fprintf(stderr, "Failed to launch align_filter(error code %s)!\n", cudaGetErrorString(err)); exit(EXIT_FAILURE);}
  
  cudaMemcpy(filter_align, d_B, fH*fW*fD*sizeof(float), cudaMemcpyDeviceToHost);
  ///////align filter end

  ///////Convolve begin (FFT, Pointwise prodcut, IFFT)
  int N[3] = {H, W, D};
  size_t real_size = W * H * D* sizeof(cufftReal);
  size_t complex_size = W * H * (D/2 + 1) * sizeof(cufftComplex);

  cudaMalloc((void**)&d_inA, real_size);
  cudaMalloc((void**)&d_inB, real_size);
  cudaMalloc((void**)&d_outA, complex_size);
  cudaMalloc((void**)&d_outB, complex_size);
  cudaMemset(d_inA,0,real_size);
  cudaMemset(d_inB,0,real_size);

  cudaMemcpy(d_inA, filter_align, real_size,  cudaMemcpyHostToDevice);
  cudaMemcpy(d_inB, input_layer_pad, real_size, cudaMemcpyHostToDevice); //update inpute_layer
  cufftPlanMany(&fwplanA, 3, N, NULL, 0,0,NULL,0,0, CUFFT_R2C ,BATCH);
  cufftPlanMany(&fwplanB, 3, N, NULL, 0,0,NULL,0,0, CUFFT_R2C ,BATCH);
  cufftPlanMany(&bwplan, 3, N, NULL, 0,0,NULL,0,0, CUFFT_C2R ,BATCH);
  
  cufftExecR2C(fwplanA, d_inA, d_outA);
  cufftExecR2C(fwplanB, d_inB, d_outB);

  int blocksx = ceil((W*H*(D/2 + 1)) / 256.0f);
  dim3 threads1(256);
  dim3 grid1(blocksx);
  pointwise_product<<<grid1, threads1>>>(d_outA, d_outB, (W*H*(D/2 + 1)), 1.0f/(H*W*D));

  cufftExecC2R(bwplan, d_outA, d_inA);
  //////convolve end

  ////////crop output
  int fH = kernel_dim[0], fW = kernel_dim[1] , fD = kernel_dim[2];
  int oH = H - fH + 1; int oW = W - fW + 1;
  float *crop_out = NULL; cudaMalloc((void **)&crop_out, oH * oW * sizeof(float));
  dim3 threads(1,1,1);
  dim3 grid(H,W,D);
  crop<<<grid1, threads1>>>(d_inA, crop_out, H, W, D, oH, oW);
  err = cudaGetLastError(); if(err!=cudaSuccess){fprintf(stderr, "Failed to launch crop(error code %s)!\n", cudaGetErrorString(err)); exit(EXIT_FAILURE);}
  ///////crop output end

  ///////stride output stride(float* f_in, float* f_out, int H, int W, int stride)
  if(stride != 1)
  {
      int sH = oH / stride + 1; int sW = oW / stride + 1; 
      float *stride_out = NULL; cudaMalloc((void **)&stride_out, sH * sW * sizeof(float));
      dim3 threads(1,1,1);
      dim3 grid(oH,oW,1);
      stride<<<grid1, threads1>>>(crop_out, stride_out ,oH, oW, stride);
      err = cudaGetLastError(); if(err!=cudaSuccess){fprintf(stderr, "Failed to launch stride(error code %s)!\n", cudaGetErrorString(err)); exit(EXIT_FAILURE);}
      oH = sH; oW = sW;
      crop_out = stride_out;
  }
  ///////stride output end

  float* result = new float[oH * oW];
  cudaMemcpy(result, crop_out , oH * oW * sizeof(float) ,cudaMemcpyDeviceToHost);
  
  return result;
  
}