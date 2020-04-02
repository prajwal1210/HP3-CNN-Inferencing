%%cuda --name /content/src/direct_convolution.cu
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>

# // whether X,Y,W are float or int

__global__ 
void direct_convolution( int C , int H , int W , int M , int K , int P, int S , int H_out , int W_out ,  int W_grid , int* X, int* W_filter, int* Y)
{
    int n , m , h , w , c , p , q ;
    n = blockIdx.x ;
    m = blockIdx.y ;
    h = blockIdx.z / W_grid + threadIdx.y;
    w = blockIdx.z % W_grid + threadIdx.x;

    H = H+P;
    W = W+P;
 
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


int main(void)
{

  cudaError_t err = cudaSuccess;
  int N,C,M,K,H,W,P,S;
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
  scanf("%d",&S);
 
  int size_input_matrix_0 = N * C * H * W * sizeof(int) ;
  int size_input_matrix = N * C * (H+P) * (W+P) * sizeof(int) ;
  int size_filter_matrix = M * C * K * K * sizeof(int) ;
  int H_out = (H - K + P + S )/S;
  int W_out = (W - K + P + S )/S;
  int size_output_matrix = N * M * H_out * W_out * sizeof(int) ;
  
  int *X   = (int*)malloc(size_input_matrix_0 );
  int *h_X = (int*)calloc(size_input_matrix/sizeof(int) , sizeof(int) );
  int *h_Y = (int*)malloc(size_output_matrix );
  int *h_W = (int*)malloc(size_filter_matrix );
 
  if (h_X == NULL || h_Y == NULL || h_W == NULL || X == NULL )
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

  int n,c,m,h,w;
 
  printf("\n Enter the input matrix : \n");
  for(n = 0 ; n < N ; n++ )
  {
      for(c = 0 ; c < C ; c++ )
      {
          for( h = 0 ; h < H ; h++ )
          {
              for( w = 0 ; w < W ; w++)
              {
                  scanf("%d",&X[ n*(C*H*W) + c*(H*W) + h*(W) + w] );
              }
          }
      }
  }
 
 for(n = 0 ; n < N ; n++ )
  {
      for(c = 0 ; c < C ; c++ )
      {
          for( h = P/2 ; h < H+P/2 ; h++ )
          {
              for( w = P/2 ; w < W+P/2 ; w++)
              {
                  h_X[ n*(C*(H+P)*(W+P)) + c*((H+P)*(W+P)) + h*(W+P) + w] =  X[ n*(C*H*W) + c*(H*W) + (h-P/2)*(W) + (w-P/2)];
              }
          }
      }
  }




 for(n = 0 ; n < N ; n++ )
  {
      printf("n = %d\n",n);
      for(c = 0 ; c < C ; c++ )
      {
          printf(" channel - %d\n",c);
          for( h = 0 ; h < H+P ; h++ )
          {
              for( w = 0 ; w < W+P ; w++)
              {
                  printf("%d ",h_X[ n*(C*(H+P)*(W+P)) + c*((H+P)*(W+P)) + h*(W+P) + w] );
              }
           printf("\n");
          }
      }
  }
 
  printf("Enter the filter matrix : \n");
  for(m = 0 ; m < M ; m++ )
  {
      for(c = 0 ; c < C ; c++ )
      {
          for( h = 0 ; h < K ; h++ )
          {
              for( w = 0 ; w < K ; w++)
              {
                  scanf("%d",&h_W[ m*(C*K*K) + c*(K*K) + h*(K) + w] );
              }
          }
      }
  }
  
  int *d_X, *d_Y, *d_W;
 
  # // copying h_X to device 
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
 
#  // copying h_W to device
 
  err = cudaMalloc((void**)&d_W, size_filter_matrix );
  if (err != cudaSuccess)
    {
      fprintf(stderr, "Failed to allocate device vector d_W (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
  
  err = cudaMemcpy( d_W , h_W , size_filter_matrix, cudaMemcpyHostToDevice );
  if (err != cudaSuccess)
    {
      fprintf(stderr, "Failed to copy vector h_W from host to device (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
 
#  // allocating memory for d_Y
 
  err = cudaMalloc((void**)&d_Y, size_output_matrix );
  if (err != cudaSuccess)
    {
      fprintf(stderr, "Failed to allocate device vector d_Y (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
  
  int tile_width = 1 , tile_height = 1;
  int w_grid = W_out / tile_width ;
  int h_grid = H_out / tile_height ;
  
  dim3 grid( N , M , w_grid * h_grid);
  dim3 block( tile_height , tile_width , 1 );
 
#  // considering W = H  =>  W_grid = H_grid  
  direct_convolution<<< grid, block >>>( C, H , W , M , K , P , S ,  H_out , W_out , w_grid , d_X , d_W , d_Y) ;

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
 
  for(n = 0 ; n < N ; n++ )
  {
      printf("N : %d \n",n);
      for(m = 0 ; m < M ; m++ )
      {
          printf("filter no. : %d \n", m);
          for( h = 0 ; h < H_out ; h++ )
          {
              for( w = 0 ; w < W_out ; w++)
              {
                  printf("%d ",h_Y[ n*(M * H_out * W_out) + m*(H_out * W_out) + h*(W_out) + w] );
              }
            printf("\n");
          }
      }
  }
 
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
