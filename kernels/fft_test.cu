#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <"fft.cu">

int main(){
    int channels = 3;
    int height = 5;
    int widht = 5;
    int kernel_height = 3;
    int kernel_width = 3;
    float input_layer[channel][height][width]  = {{{0,0,1,0,2},
      {1,0,2,0,1},
      {1,0,2,2,0},
      {2,0,0,2,0},
      {2,1,2,2,0}},{{2,1,2,1,1},
      {2,1,2,0,1},
      {0,2,1,0,1},
      {1,2,2,2,2},
      {0,1,2,0,1}},{{2,1,1,2,0},
      {1,0,0,1,0},
      {0,1,0,0,0},
      {1,0,2,1,0},
      {2,2,1,1,1}}};
      
    float kernel[channel][kernel_height][kernel_width] = {{{-1,0,1},
      {0,0,1},
      {1,-1,1}},{{-1,0,1},
      {1,-1,1},
      {0,1,0}},{{-1,1,1},
      {1,1,0},
      {0,-1,0}}};
      
    int pad = 0;
    int stride = 2;
    int batch_size = 1;
    int il_dim[3] = {height, width, channel}; int kernel_dim[3] = {kernel_height, kernel_width, channel};
    float actual_result[height][width];
    float expected_result[height][width] = {{2,3,3},{3,7,3},{8,10,-3}};
    
    //actual_result = convolve_FFT(input_layer, kernel, pad, stride, batch_size, il_dim, kernel_dim);
    for(int i = 0; i < H; i++)
      {
          for(int j = 0; j < W; j++)
          {
              if(abs(actual_result[i][j] - expected_result[i][j]) > 0.001 ){
                  printf("convolution failed \n expected element: %f \n actual element: %f \n", expected_result[i][j],actual_result[i][j])
              }
          }
      }
      printf("Test Passed.\n");
    return 0;
}
