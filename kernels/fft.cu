#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>

//multiplication of 2 cufftComplex elements
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


// input arguments are input_layer, kernel, padding, stride, batch_size, input_layer dimensions, kernel dimensions

float* convolve_FFT(float * input_layer, float * kernel, int padding, int stride, int batch_size, int* il_dim, int* kernel_dim)
{
  int W = 5;
  int H = 1;
  int D = 5;
  
  int BATCH = 1;
 
  float B[H][W][D] = {{
      {0,     0,    0,    0,  0},
      {0,     1,    2,    1,  0},
      {0,     2,    1,    2,  0},
      {0,     1,    2,    1,  0},
      {0,     0,    0,    0,  0}}};
 
  
  float old_filter[H][W][D] = {{
      {0,     0,    0,    0,  0},
      {0,     0,    1,    0,  0},
      {0,     1,    1,    1,  0},
      {0,     0,    1,    0,  0},
      {0,     0,    0,    0,  0}}};
 
  float *A = (float *)malloc(H*W*D*sizeof(float));
  float *d_A = NULL;
  cudaMalloc((void **)&d_A, H*W*D*sizeof(float));
 
  float *d_B = NULL;
  cudaMalloc((void **)&d_B, H*W*D*sizeof(float));
 
  cudaMemcpy(d_A, old_filter, H*W*D*sizeof(float), cudaMemcpyHostToDevice);
 
  dim3 threads(1,1,1);
  dim3 grid(H,W,D);
  cudaError_t err = cudaSuccess;
  align_filer<<<grid, threads>>>(d_A, d_B, H,W,D);
  err = cudaGetLastError();
  if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
  
  cudaMemcpy(A, d_B, H*W*D*sizeof(float), cudaMemcpyDeviceToHost);

  cufftReal* d_inA, *d_inB;
  cufftComplex* d_outA, *d_outB;

  size_t real_size = W * H * D* sizeof(cufftReal);
  size_t complex_size = W * H * (D/2 + 1) * sizeof(cufftComplex);

  cudaMalloc((void**)&d_inA, real_size);
  cudaMalloc((void**)&d_inB, real_size);

  cudaMalloc((void**)&d_outA, complex_size);
  cudaMalloc((void**)&d_outB, complex_size);

  cudaMemset(d_inA,0,real_size);
  cudaMemset(d_inB,0,real_size);

  cudaMemcpy(d_inA, A, real_size,	cudaMemcpyHostToDevice);
  cudaMemcpy(d_inB, B, real_size,	cudaMemcpyHostToDevice);

  cufftHandle fwplanA, fwplanB, bwplan;

  int N[3] = {H, W, D};
  cufftPlanMany(&fwplanA, 3, N, NULL, 0,0,NULL,0,0, CUFFT_R2C ,BATCH);
  cufftPlanMany(&fwplanB, 3, N, NULL, 0,0,NULL,0,0, CUFFT_R2C ,BATCH);
  cufftPlanMany(&bwplan, 3, N, NULL, 0,0,NULL,0,0, CUFFT_C2R ,BATCH);

  cufftExecR2C(fwplanA, d_inA, d_outA);
  cufftExecR2C(fwplanB, d_inB, d_outB);

  int blocksx = ceil((W*H*(D/2 + 1)) / 256.0f);
  dim3 threads1(256);
  dim3 grid1(blocksx);
  
  // One complex product for each thread scaled by total elements
  pointwise_product<<<grid1, threads1>>>(d_outA, d_outB, (W*H*(D/2 + 1)), 1.0f/(H*W*D));

  cufftExecC2R(bwplan, d_outA, d_inA);


  float* result = new cufftReal[W*H*2*(D/2+1)];
  cudaMemcpy(result, d_inA, real_size,cudaMemcpyDeviceToHost);
  
  return result;
  
}