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

//flip filter about the center element
// <<H,W,D>>
__global__ void flip_filer(float* f_in, float* f_out, int H, int W, int D)
{
    int col = blockIdx.x*blockDim.x+threadIdx.x;
    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int dep = blockIdx.z*blockDim.z+threadIdx.z;
    int i = dep + col * D + row * H * D;
 
    int new_col = H - col -1;
    int new_row = W - row - 1;
 
    int j = dep + new_col * D + new_row * H * D;

    if(col < H && row < W && dep < D)
    {
        f_out[j] = f_in[i];
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
    int i = dep + col * D + row * H * D;
 
    int new_col = ((col - H/2) % H);
    int new_row = ((row - W/2) % W);
    int new_dep = ((dep - D/2) % D);
 
    new_col = new_col < 0 ? H + new_col: new_col;
    new_row = new_row < 0 ? W + new_row: new_row;
    new_dep = new_dep < 0 ? D + new_dep: new_dep;

    int j = dep + new_col * D + new_row * H * D;

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

    int new_H = H+2*pad; int new_W = W+2*pad; 
 
    int i = dep + col * D + row * new_H * D;
    int j = dep + (col - pad) *D+ (row - pad) * H * D ;

    if(col < new_H && row < new_W && dep < D)
    {
        if((col < pad || col > H+pad-1) || (row < pad || row > W+pad-1)) f_out[i] = 0;
        else f_out[i] = f_in[j];
    }
}

//crop and output the required size output (O = ((W - k + 2 * P)) + 1)
//<<H, W, D>>
__global__ void crop(float* f_in, float* f_out, int H, int W, int O_H, int O_W)
{
    int col = blockIdx.x*blockDim.x+threadIdx.x;
    int row = blockIdx.y*blockDim.y+threadIdx.y;
    //int dep = blockIdx.z*blockDim.z+threadIdx.z;
    int i =  col + row * H ;
    
    int crop_H = (H - O_H)/2;
    int crop_W = (W - O_W)/2;

    int j = (col - crop_H) + (row - crop_W) * (O_H);

    if(col < H && row < W)
    {
        if(col >= crop_H && col < H - crop_H && row >= crop_W && row < W - crop_W)f_out[j] = f_in[i];
    }
}

//stride and output the required size output (O = ((W - k + 2 * P)/stride) + 1)
//call this only if stride is not 1
//<<H, W, 1>>
__global__ void stride_(float* f_in, float* f_out, int H, int W, int stride)
{
    int col = blockIdx.x*blockDim.x+threadIdx.x;
    int row = blockIdx.y*blockDim.y+threadIdx.y;
    //int dep = blockIdx.z*blockDim.z+threadIdx.z;
 
    int O_H = H/stride + 1;
    int O_W = W/stride + 1;
    
    int i = col * W + row; 
    //+ dep * H * W;

    if(col < H && row < W && (col%stride == 0) && (row%stride == 0))
    {
        int j = (col/stride) * O_W + (row/stride) ;
        // dep * H * W;
        f_out[j] = f_in[i];
    }
}

float* conv_operation(float* filter_align, float* input_layer_pad, int H, int W, int D)
{
    int N[3] = {H, W, D};
    cufftReal* d_inA, *d_inB;
    cufftComplex* d_outA, *d_outB;
    cufftHandle fwplanA, fwplanB, bwplan;
    size_t real_size = W * H * sizeof(cufftReal);
    size_t complex_size = W * (H/2 + 1) * sizeof(cufftComplex);

    cudaMalloc((void**)&d_inA, real_size);
    cudaMalloc((void**)&d_inB, real_size);
    cudaMalloc((void**)&d_outA, complex_size);
    cudaMalloc((void**)&d_outB, complex_size);
    cudaMemset(d_inA,0,real_size);
    cudaMemset(d_inB,0,real_size);

    cudaMemcpy(d_inA, filter_align, real_size,  cudaMemcpyHostToDevice);
    cudaMemcpy(d_inB, input_layer_pad, real_size, cudaMemcpyHostToDevice); //update inpute_layer
  
    cufftPlan2d(&fwplanA, H, W, CUFFT_R2C);
    cufftPlan2d(&fwplanB,  H, W, CUFFT_R2C);
    cufftPlan2d(&bwplan, H, W, CUFFT_C2R);

    
    //cufftPlanMany(&fwplanA, 2, N, NULL, 0,0,NULL,0,0, CUFFT_R2C ,D);
    //cufftPlanMany(&fwplanB, 2, N, NULL, 0,0,NULL,0,0, CUFFT_R2C ,D);
    //cufftPlanMany(&bwplan, 2, N, NULL, 0,0,NULL,0,0, CUFFT_C2R ,D);
    
    cufftExecR2C(fwplanA, d_inA, d_outA);
    cufftExecR2C(fwplanB, d_inB, d_outB);

    int blocksx = ceil((W*(H/2 + 1)) / 256.0f);
    dim3 threads4(256);
    dim3 grid4(blocksx);
    pointwise_product<<<grid4, threads4>>>(d_outA, d_outB, (W*(H/2 + 1)), 1.0f/(H*W));

    cufftExecC2R(bwplan, d_outA, d_inA);
    
    float* result1 = new float[W*2*((H/2 + 1)) ];
    cudaMemcpy(result1, d_inA, real_size,cudaMemcpyDeviceToHost);
    return result1;
}

float* add_filters(float* input1, float* input2, int size)
{
    float* output = (float*)malloc(size*sizeof(float));
    for(int i = 0; i < size; i++)
    {
        output[i] = input1[i] + input2[i];
    }
    return output;
}

// input arguments are input_layer, kernel, padding, stride, batch_size, input_layer dimensions, kernel dimensions
// Operations: pad(input), pad(filter), align(filter), output = convolve(input,filter), crop(output), stride(output)  

float* convolve_FFT(float * input_layer, float * kernel, int pad, int stride, int batch_size, int* il_dim, int* kernel_dim)
{
  //////initializations
  int H = il_dim[0], W = il_dim[1], D = il_dim[2];
  int fH = kernel_dim[0], fW = kernel_dim[1] , fD = kernel_dim[2];
  cudaError_t err = cudaSuccess;
  //////////////////////////

  ///////pad input
  int new_H = H+2*pad; int new_W = W+2*pad;
  float *pad_input_in = NULL; cudaMalloc((void **)&pad_input_in, H * W * D * sizeof(float));
  float *pad_input_out = NULL; cudaMalloc((void **)&pad_input_out, new_H * new_W * D * sizeof(float));
  float *input_layer_pad = (float *)malloc(new_H* new_W * D *sizeof(float));

  cudaMemcpy(pad_input_in, input_layer, H * W * D * sizeof(float) , cudaMemcpyHostToDevice);
  dim3 threads1(1,1,1);
  dim3 grid1(new_H,new_W,D);
  pad_input<<<grid1,threads1>>>(pad_input_in, pad_input_out, H,W,D,pad);
  err = cudaGetLastError(); if(err!=cudaSuccess){fprintf(stderr, "Failed to launch pad input (error code %s)!\n", cudaGetErrorString(err)); exit(EXIT_FAILURE);}
  cudaMemcpy(input_layer_pad, pad_input_out , new_H * new_W * D * sizeof(float), cudaMemcpyDeviceToHost);
  H = new_H; W = new_W;
  //////pad input end
 

  //////flip filter
  float *filter_flip = (float *)malloc(fH * fW * fD *sizeof(float));
  float *f_A = NULL; cudaMalloc((void **)&f_A, fH * fW * fD * sizeof(float));
  float *f_B = NULL; cudaMalloc((void **)&f_B, fH * fW * fD * sizeof(float));
  cudaMemcpy(f_A, kernel , fH * fW * fD * sizeof(float), cudaMemcpyHostToDevice);

  dim3 threads0(1,1,1);
  dim3 grid0(fH,fW,fD);
  flip_filer<<<grid0, threads0>>>(f_A, f_B, fH,fW,fD);
  err = cudaGetLastError(); if(err!=cudaSuccess){fprintf(stderr, "Failed to launch align_filter(error code %s)!\n", cudaGetErrorString(err)); exit(EXIT_FAILURE);}
  
  cudaMemcpy(filter_flip, f_B, fH*fW*fD*sizeof(float), cudaMemcpyDeviceToHost);

 ///////flip filter end


  ///////pad filter 
  int fpad = (new_H - fH)/2; 
  int new_fH = fH+2*fpad; int new_fW = fW+2*fpad;
  float *pad_filter_in = NULL; cudaMalloc((void **)&pad_filter_in, fH * fW * fD * sizeof(float));
  float *pad_filter_out = NULL; cudaMalloc((void **)&pad_filter_out, new_fH * new_fW * D * sizeof(float));
  float *filter_pad = (float *)malloc(new_fH* new_fW * D *sizeof(float));

  cudaMemcpy(pad_filter_in, filter_flip , fH * fW * fD * sizeof(float) , cudaMemcpyHostToDevice);
  dim3 threads2(1,1,1);
  dim3 grid2(new_fH,new_fW,D);

  pad_input<<<grid2,threads2>>>(pad_filter_in, pad_filter_out, fH,fW,D,fpad);
  err = cudaGetLastError(); if(err!=cudaSuccess){fprintf(stderr, "Failed to launch pad filter(error code %s)!\n", cudaGetErrorString(err)); exit(EXIT_FAILURE);}
  cudaMemcpy(filter_pad, pad_filter_out , new_fH * new_fW * D * sizeof(float), cudaMemcpyDeviceToHost);
  fH = new_fH; fW = new_fW;
  //////pad filter end
 for(int i = 0; i < fD; i++)
    {
      for(int j = 0; j < fH; j++)
      {
          for(int k = 0; k < fW; k++)
          {
              printf("%f ",filter_pad[i + j * fD+ k * fD * fH]  );
          }
          printf("\n");
      }    
      printf("\n");
   } 
  printf("\n\n");
  ///////align filter begin
  float *filter_align = (float *)malloc(fH * fW * fD *sizeof(float));
  float *d_A = NULL; cudaMalloc((void **)&d_A, fH * fW * fD * sizeof(float));
  float *d_B = NULL; cudaMalloc((void **)&d_B, fH * fW * fD * sizeof(float));
  cudaMemcpy(d_A, filter_pad, fH * fW * fD * sizeof(float), cudaMemcpyHostToDevice);

  dim3 threads3(1,1,1);
  dim3 grid3(fH,fW,fD);
  align_filer<<<grid3, threads3>>>(d_A, d_B, fH,fW,fD);
  err = cudaGetLastError(); if(err!=cudaSuccess){fprintf(stderr, "Failed to launch align_filter(error code %s)!\n", cudaGetErrorString(err)); exit(EXIT_FAILURE);}
  
  cudaMemcpy(filter_align, d_B, fH*fW*fD*sizeof(float), cudaMemcpyDeviceToHost);
  ///////align filter end
 
  for(int i = 0; i < fD; i++)
    {
      for(int j = 0; j < fH; j++)
      {
          for(int k = 0; k < fW; k++)
          {
              printf("%f ",filter_align[i + j * fD+ k * fD * fH]  );
          }
          printf("\n");
      }    
      printf("\n");
   } 
  printf("\n\n");


  ///////Convolve begin (FFT, Pointwise prodcut, IFFT)
  
  float* f_im_conv = (float *)malloc(H*W*sizeof(float));
  float* f_ker_conv = (float *)malloc(H*W*sizeof(float));
  float* result1; 
  float * conv_result = (float *)malloc(H*W*sizeof(float)); for(int i = 0; i < H*W; i++){conv_result[i] = 0;}
  for(int i = 0; i < D; i++)
  {
     for(int j = 0; j < H; j++)
      {
          for(int k = 0; k < W; k++)
          {
            f_im_conv[j + k* W] = input_layer_pad[i + j * fD+ k * fD * fH];
            f_ker_conv[j + k* W] = filter_align[i + j * fD+ k * fD * fH];
          }
      }    
      result1 = conv_operation(f_im_conv, f_ker_conv, H, W, D);
      conv_result = add_filters(result1, conv_result, H*W);
      printf("result of conv %d\n", i);
        for(int j = 0; j < H; j++)
        {
            for(int k = 0; k < W; k++)
            {
                printf("%f ",result1[j + k*W]  );
            }
            printf("\n");
        }   
       printf("\n");
      printf("result of conv end %d\n\n", i);
   } 

    printf("result of conv final\n");
        for(int j = 0; j < H; j++)
        {
            for(int k = 0; k < W; k++)
            {
                printf("%f ",conv_result[j + k*W]  );
            }
            printf("\n");
        }   
       printf("\n");
   printf("result of conv final end \n\n");
  
  //////convolve end

  ////////crop output
  fH = kernel_dim[0]; fW = kernel_dim[1] ; fD = kernel_dim[2];
  int oH = H - fH + 1; int oW = W - fW + 1;
   
  //test_crop(result1,H, W, D, oH, oW);
  
  float *crop_out = NULL; cudaMalloc((void **)&crop_out, oH * oW * sizeof(float));
  float *crop_in = NULL; cudaMalloc((void **)&crop_in, H * W * sizeof(float));
  cudaMemcpy(crop_in, conv_result,  H * W * sizeof(float),cudaMemcpyHostToDevice);
  
  dim3 threads5(1,1,1);
  dim3 grid5(H,W,1);
  crop<<<grid5, threads5>>>(crop_in, crop_out, H, W, oH, oW);
  err = cudaGetLastError(); if(err!=cudaSuccess){fprintf(stderr, "Failed to launch crop(error code %s)!\n", cudaGetErrorString(err)); exit(EXIT_FAILURE);}
  
  float* result2 = (float*)malloc(oW*oH* sizeof(float));
  cudaMemcpy(result2, crop_out, oW*oH* sizeof(float) ,cudaMemcpyDeviceToHost);
  ///////crop output end
      
      for(int j = 0; j < oH; j++)
      {
          for(int k = 0; k < oW; k++)
          {
              printf("%f ",result2[ j+ k * oH]  );
          }
          printf("\n");
      } 
      printf("\n\n\n");

  ///////stride output stride_(float* f_in, float* f_out, int H, int W, int stride)
  if(stride != 1)
  {
      int sH = oH / stride + 1; int sW = oW / stride + 1; 
      float *stride_in = NULL; cudaMalloc((void **)&stride_in, oH * oW * sizeof(float));
      float *stride_out = NULL; cudaMalloc((void **)&stride_out, sH * sW * sizeof(float));
      cudaMemcpy(stride_in, result2, oW*oH* sizeof(float) ,cudaMemcpyHostToDevice);
      dim3 threads6(1,1,1);
      dim3 grid6(oH,oW,1);
      stride_<<<grid6, threads6>>>(stride_in, stride_out ,oH, oW, stride);
      err = cudaGetLastError(); if(err!=cudaSuccess){fprintf(stderr, "Failed to launch stride(error code %s)!\n", cudaGetErrorString(err)); exit(EXIT_FAILURE);}
      oH = sH; oW = sW;
      crop_out = stride_out;
  }
  ///////stride output end

  float* result = new float[oH * oW];
  cudaMemcpy(result, crop_out , oH * oW * sizeof(float) ,cudaMemcpyDeviceToHost);
  
  return result;
  
}