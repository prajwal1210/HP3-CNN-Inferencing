//%%cuda --name /content/src/direct_convolution.cu

/*including the required library*/
#include "direct_conv.hpp"
using namespace std;
/*parallelization code */

__global__ 
void direct_convolution(int input_channels, int input_height, int input_width, int out_channels, int kernel_height,int kernel_width, int padding, int stride, int H_out, int W_out, int W_grid, int tile_w, float* X, float* W_filter, float* Y)
{
    int n , m , h , w , c , p , q ;
    n = blockIdx.x ;
    m = blockIdx.y ;
    h = (blockIdx.z / W_grid)*tile_w + threadIdx.y;
    w = (blockIdx.z % W_grid)*tile_w + threadIdx.x;

    input_height = input_height+padding;
    input_width = input_width+padding;

    if(h<H_out && w<W_out)
    {
        int temp=0;
        for( c = 0 ; c < input_channels ; c++ )
        {
            for( p = 0 ; p < kernel_height ; p++ )
            {
                for( q = 0 ; q < kernel_width ; q++ )
                    temp = temp + X[ n*(input_channels*input_height*input_width) + c*(input_height*input_width) + (h*stride+p)*(input_width) + (w*stride+q)] * W_filter[ m*(input_channels*kernel_height*kernel_width) + c*(kernel_height*kernel_width) + p*(kernel_height) + q];
            }
        }    
        Y[n*(out_channels*H_out*W_out) + m*(H_out*W_out) + h*(W_out) + w] = temp;
    }
}

/*forward pass function declared in direc_conv.hpp library*/
float *passforward(int out_channels, int input_channels, int kernel_height, int kernel_width, int padding, int stride, float* kernel_weights,int batchsize_of_data, int input_height, int input_width, float* input)
{
  if(kernel_height > input_height || kernel_width > input_width){
      cout << "kernel size is too big " << endl;
      exit(EXIT_FAILURE);
  }
  cudaError_t err = cudaSuccess;
 
  /* size of matrix with padding*/ 
  int size_input_matrix = batchsize_of_data * input_channels * (input_height+padding) * (input_width+padding) * sizeof(float) ;   // size of input matrix after padding
 
  /*size of filter matrix*/
  int size_filter_matrix = out_channels * input_channels * kernel_width * kernel_width * sizeof(float) ; 
  
  /* calculating size of output matrix*/
  int H_out = (input_height - kernel_height + padding + stride )/stride;
  int W_out = (input_width - kernel_width + padding + stride )/stride;
  int size_output_matrix = batchsize_of_data * out_channels * H_out * W_out * sizeof(float) ;
  
  /*allocating memory for input  matrix with padding*/
  float *h_X = (float*)calloc(size_input_matrix/sizeof(float) , sizeof(float) );  

  /*allocating memory for output matrix*/
  float *h_Y = (float*)malloc(size_output_matrix );             
 
  /* memory allocation check*/
  if (h_X == NULL || h_Y == NULL )
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

  int n,c,m,h,w;

  /*creating input matrix with padding value zero */
  for(n = 0 ; n < batchsize_of_data ; n++ )
  {
      for(c = 0 ; c < input_channels ; c++ )
      {
          for( h = padding/2 ; h < input_height+padding/2 ; h++ )
          {
              for( w = padding/2 ; w < input_width+padding/2 ; w++)
              {
                  h_X[ n*(input_channels*(input_height+padding)*(input_width+padding)) + c*((input_height+padding)*(input_width+padding)) + h*(input_width+padding) + w] =  input[ n*(input_channels*input_height*input_width) + c*(input_height*input_width) + (h-padding/2)*(input_width) + (w-padding/2)];
              }
          }
      }
  } 

  float *d_X, *d_Y, *d_W; 
 
  /*allocating memory for padded matrix in the device*/
 
  err = cudaMalloc((void**)&d_X, size_input_matrix );
  if (err != cudaSuccess)
    {
      fprintf(stderr, "Failed to allocate device vector d_X (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
  
  /* copying padded matrix to device */

  err = cudaMemcpy( d_X , h_X , size_input_matrix , cudaMemcpyHostToDevice );
  if (err != cudaSuccess)
    {
      fprintf(stderr, "Failed to copy vector h_X from host to device (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
 
  /*allocating memory for kernel weights*/
 
  err = cudaMalloc((void**)&d_W, size_filter_matrix );
  if (err != cudaSuccess)
    {
      fprintf(stderr, "Failed to allocate device vector d_W (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
  
  /*copying kernel weights matrix*/
  err = cudaMemcpy( d_W , kernel_weights , size_filter_matrix, cudaMemcpyHostToDevice );
  if (err != cudaSuccess)
    {
      fprintf(stderr, "Failed to copy vector h_W from host to device (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
 
  /*allocating memory for the output matrix*/
 
  err = cudaMalloc((void**)&d_Y, size_output_matrix );
  if (err != cudaSuccess)
    {
      fprintf(stderr, "Failed to allocate device vector d_Y (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
  
  
  /* making sure that 1024 threads isn't crossed*/
  int tile_width = 2 , tile_height = 2;   
  int w_grid = ceil((W_out*1.0) / tile_width) ;
  int h_grid = ceil((H_out*1.0) / tile_height) ;
  
 
  int temp  = w_grid * h_grid;
  dim3 grid( batchsize_of_data , out_channels , temp );
  dim3 block( tile_width , tile_height , 1 );
 

  /* calling the direct_convolution kernel */  
  direct_convolution<<< grid, block >>>( input_channels, input_height , input_width , out_channels ,kernel_height, kernel_width , padding , stride ,  H_out , W_out , w_grid ,  tile_width ,  d_X , d_W , d_Y) ;

  err = cudaGetLastError();

  /*checking if the device program is executed or not*/
  if (err != cudaSuccess)
  {
    fprintf(stderr, "Failed to launch reduce1 kernel (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
 
  /*copying output matrix into host and checking*/
  err = cudaMemcpy(h_Y, d_Y, size_output_matrix , cudaMemcpyDeviceToHost);
  if (err != cudaSuccess)
  {
    fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  
   /* releasing all the device and host vectors */
 
  err = cudaFree(d_X);
  if (err != cudaSuccess)
  {
    fprintf(stderr, "Failed to free device vector X (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }      
  err = cudaFree(d_Y);
  if (err != cudaSuccess)
  {
    fprintf(stderr, "Failed to free device vector Y (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  err = cudaFree(d_W);
  if (err != cudaSuccess)
  {
    fprintf(stderr, "Failed to free device vector W (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }  
 
  /*releasing the memory*/
  free(h_X);
  return h_Y;      
}