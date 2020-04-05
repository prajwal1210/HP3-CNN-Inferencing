

//%%cuda --name /content/src/direct_convolution.cu


#include "direct_conv.h"

/*# /*
# N - batch size of the input
# C - channel size of input
# K - kernel height = kernel width
# M - no of output feature maps( output channel size )
# H - height of the input data
# W - width of the input data
# P - padding ( heigth  = width ) , P/2 units added on the either sides
# S - stride ( height = width) , S units in one direction
# H_out - height of output data
# W_out  - weight ofoutput data

# - each thread will compute one element of one output feature map

# */

__global__ 
void direct_convolution( int C , int H , int W , int M , int K , int P, int S , int H_out ,
 int W_out ,  int W_grid  , int tile_w  , float* X, float* W_filter, float* Y)
{
    int n , m , h , w , c , p , q ;
    n = blockIdx.x ;
    m = blockIdx.y ;
    h = (blockIdx.z / W_grid)*tile_w + threadIdx.y;
    w = (blockIdx.z % W_grid)*tile_w + threadIdx.x;

    H = H+P;
    W = W+P;

    if(h<H_out && w<W_out)
    {
        int temp=0;
        for( c = 0 ; c < C ; c++ )
        {
            for( p = 0 ; p < K ; p++ )
            {
                for( q = 0 ; q < K ; q++ )
                {
                    temp = temp + X[ n*(C*H*W) + c*(H*W) + (h*S+p)*(W) + (w*S+q)] * W_filter[ m*(C*K*K) + c*(K*K) + p*(K) + q] ;
                }
            }
        }
    
        Y[n*(M*H_out*W_out) + m*(H_out*W_out) + h*(W_out) + w] = temp;

    }


    }


void passforward(int out_channels, int input_channels, int kernel_height, int kernel_width, int padding, int stride, float* kernel_weights
		int batchsize_of_data, int input_height, int input_width, float* input);

{

  cudaError_t err = cudaSuccess;
 
  /*int N,C,M,K,H,W,P,S;
  printf("Enter the batch size : ");
  scanf("%d",&N);
  printf("\nEnter the channel size : ");
  scanf("%d",&C);
  printf("\nEnter the height of input feature maps : ");
  scanf("%d",&H);
  printf("\nEnter the width of input feature maps : ");
  scanf("%d",&W);
  printf("\nEnter the number of output feature maps : ");
  scanf("%d",&M);
  printf("\nEnter the kernel width(same as height) :  ");
  scanf("%d",&K);
  printf("\nEnter the padding size : ");
  scanf("%d",&P);
  printf("\nEnter the striding size : ");
  scanf("%d",&S);*/
 
  int size_input_matrix_0 = batchsize_of_data * input_channels * input_height * input_width * sizeof(float) ;         // size of input matrix before padding
 
  int size_input_matrix = batchsize_of_data * input_channels * (input_height+padding) * (input_width+padding) * sizeof(float) ;   // size of input matrix after padding
 
  int size_filter_matrix = out_channels * input_channels * kernel_width * kernel_width * sizeof(float) ; 
 
  int H_out = (input_height - kernel_width + padding + stride )/stride;
  int W_out = (input_width - kernel_width + padding + stride )/stride;
  int size_output_matrix = batchsize_of_data * out_channels * H_out * W_out * sizeof(float) ;
  
  //float *X   = (float*)malloc(size_input_matrix_0 );     // X is input data matrix
  float *h_X = (float*)calloc(size_input_matrix/sizeof(float) , sizeof(float) );  // h_X is input data matrix after padding, this is sent to device for computation
  float *h_Y = (float*)malloc(size_output_matrix );                 // h_Y is the output data matrix
  //float *h_W = (float*)malloc(size_filter_matrix );                 // h_W is the filter weights matrix
 
  if (h_X == NULL || h_Y == NULL || h_W == NULL || X == NULL )
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

  int n,c,m,h,w;
         
  /*            #  /*  reading the input data matrix from user */
 
  /*printf("\n Enter the input matrix : \n");
  for(n = 0 ; n < batchsize_of_data ; n++ )
  {
      for(c = 0 ; c < input_channels ; c++ )
      {
          for( h = 0 ; h < input_height ; h++ )
          {
              for( w = 0 ; w < input_width ; w++)
              {
                  scanf("%f",&X[ n*(input_channels*input_height*input_width) + c*(input_height*input_width) + h*(input_width) + w] );
              }
          }
      }
  }*/
 
            /*    # /* padding the input data matrix with zeroes */
 
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

	            /*    # /* printing the input data matrix after padding */

/* for(n = 0 ; n < batchsize_of_data ; n++ )
  {
      printf("n = %d\n",n);
      for(c = 0 ; c < input_channels ; c++ )
      {
          printf(" channel - %d\n",c);
          for( h = 0 ; h < input_height+P ; h++ )
          {
              for( w = 0 ; w < input_width+P ; w++)
              {
                  printf("%f ",h_X[ n*(input_channels*(input_height+P)*(input_width+P)) + c*((input_height+P)*(input_width+P)) + h*(input_width+P) + w] );
              }
           printf("\n");
          }
      }
  }*/
 
            /*    # /* reading the filter weights matrix from user */
 
/*  printf("Enter the filter matrix : \n");
  for(m = 0 ; m < out_channels ; m++ )
  {
      for(c = 0 ; c < input_channels ; c++ )
      {
          for( h = 0 ; h < kernel_width ; h++ )
          {
              for( w = 0 ; w < kernel_width ; w++)
              {
                  scanf("%f",&h_W[ m*(input_channels*kernel_width*kernel_width) + c*(kernel_width*kernel_width) + h*(kernel_width) + w] );
              }
          }
      }
  }*/
  

  float *d_X, *d_Y, *d_W;  // device vectors : d_X - input matrix , d_Y - output matrix , d_W - filter weights matix
 
                  # /* copying h_X to device */
 
  err = cudaMalloc((void**)&d_X, size_input_matrix );
  if (err != cudaSuccess)
    {
      fprintf(stderr, "Failed to allocate device vector d_X (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
  
  err = cudaMemcpy( d_X , h_X , size_input_matrix , cudaMemcpyHostToDevice );
  if (err != cudaSuccess)
    {
      fprintf(stderr, "Failed to copy vector h_X from host to device (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
 
                    #  /* copying h_W to device */
 
  err = cudaMalloc((void**)&d_W, size_filter_matrix );
  if (err != cudaSuccess)
    {
      fprintf(stderr, "Failed to allocate device vector d_W (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
  
  err = cudaMemcpy( d_W , kernel_weights , size_filter_matrix, cudaMemcpyHostToDevice );
  if (err != cudaSuccess)
    {
      fprintf(stderr, "Failed to copy vector h_W from host to device (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
 
                      # /* allocating memory for d_Y */
 
  err = cudaMalloc((void**)&d_Y, size_output_matrix );
  if (err != cudaSuccess)
    {
      fprintf(stderr, "Failed to allocate device vector d_Y (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
  
    

  int tile_width = 2 , tile_height = 2;   
  int w_grid = ceil((W_out*1.0) / tile_width) ;
  int h_grid = ceil((H_out*1.0) / tile_height) ;
  
 
  int temp  = w_grid * h_grid;
  dim3 grid( batchsize_of_data , out_channels , temp );
  dim3 block( tile_width , tile_height , 1 );
 

          # /* calling the direct_convolution kernel */  
  direct_convolution<<< grid, block >>>( input_channels, input_height , input_width , out_channels , kernel_width , padding , stride ,  H_out , W_out , w_grid ,  tile_width ,  d_X , d_W , d_Y) ;



  err = cudaGetLastError();

  if (err != cudaSuccess)
  {
    fprintf(stderr, "Failed to launch reduce1 kernel (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
 
  err = cudaMemcpy(h_Y, d_Y, size_output_matrix , cudaMemcpyDeviceToHost);
  if (err != cudaSuccess)
  {
    fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
 
              # /* printing the output matrix */
  for(n = 0 ; n < batchsize_of_data ; n++ )
  {
      printf("N : %d \n",n);
      for(m = 0 ; m < out_channels ; m++ )
      {
          printf("filter no. : %d \n", m);
          for( h = 0 ; h < H_out ; h++ )
          {
              for( w = 0 ; w < W_out ; w++)
              {
                  printf("%f ",h_Y[ n*(out_channels * H_out * W_out) + m*(H_out * W_out) + h*(W_out) + w] );
              }
            printf("\n");
          }
      }
  }
 
          #  /* releasing all the device and host vectors */
 
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
 
  free(h_X);
  free(h_W);
  free(h_Y);      

  
  return 0;
}
