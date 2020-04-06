#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <"fft.cu">

int main(){
    int channel = 3;
    int height = 5;
    int width = 5;
    int kernel_height = 3;
    int kernel_width = 3;
    int batch_size = 3;
    int pad = 0;
    int stride = 2;
    int out_size = 2;
    int il_dim[3] = {height, width, channel}; int kernel_dim[3] = {kernel_height, kernel_width, channel};

    float input_layer_tmp[batch_size][channel][height][width]  = {
       {
          { {0,0,1,0,2}, {1,0,2,0,1}, {1,0,2,2,0}, {2,0,0,2,0},{2,1,2,2,0} },
          { {2,1,2,1,1}, {2,1,2,0,1}, {0,2,1,0,1}, {1,2,2,2,2}, {0,1,2,0,1}},
          { {2,1,1,2,0}, {1,0,0,1,0}, {0,1,0,0,0}, {1,0,2,1,0}, {2,2,1,1,1}}
      },
      {
          { {0,0,1,0,2}, {1,0,2,0,1}, {1,0,2,2,0}, {2,0,0,2,0},{2,1,2,2,0} },
          { {2,1,2,1,1}, {2,1,2,0,1}, {0,2,1,0,1}, {1,2,2,2,2}, {0,1,2,0,1}},
          { {2,1,1,2,0}, {1,0,0,1,0}, {0,1,0,0,0}, {1,0,2,1,0}, {2,2,1,1,1}}
      },

      {
          { {0,0,1,0,2}, {1,0,2,0,1}, {1,0,2,2,0}, {2,0,0,2,0},{2,1,2,2,0} },
          { {2,1,2,1,1}, {2,1,2,0,1}, {0,2,1,0,1}, {1,2,2,2,2}, {0,1,2,0,1}},
          { {2,1,1,2,0}, {1,0,0,1,0}, {0,1,0,0,0}, {1,0,2,1,0}, {2,2,1,1,1}}
       }
      
    };
    
    float* input_layer = (float *)malloc(batch_size* channel * height* width * sizeof(float));
    float* kernel = (float *)malloc(channel * kernel_height* kernel_width * sizeof(float));  
    float* final_output = (float *)malloc(batch_size * out_size * height * width * sizeof(float));

    float kernel_tmp[out_size][channel][kernel_height][kernel_width] = 
     {
        {
            {{-1,0,1}, {0,0,1}, {1,-1,1}},
            {{-1,0,1}, {1,-1,1}, {0,1,0}},
            {{-1,1,1}, {1,1,0}, {0,-1,0}}
        },
        {
            {{-1,0,1}, {0,0,1}, {1,-1,1}},
            {{-1,0,1}, {1,0,1}, {0,1,0}},
            {{-1,1,1}, {1,1,0}, {0,-1,0}}
        }
     };

    for(int l = 0; l < batch_size; l++)
    {
      for(int i = 0; i < channel; i++)
      {
        for(int j = 0; j < height; j++)
        {
            for(int k = 0; k < width; k++)
            {
                input_layer[l * channel * height * width + i * height * width + j * width + k] = input_layer_tmp[l][i][j][k];  
              //printf("%f ",  input_layer[i * height * width + j * width + k]);
            }
          //printf("\n");
        }    
        //printf("\n");
      }
    }
  int out_H = ((height - kernel_height + 2 * pad)/stride) + 1; 
  int out_W = ((width - kernel_width + 2 * pad)/stride) + 1; 
   for(int l = 0; l < out_size ; l++)
   {
        for(int i = 0; i < channel; i++)
        {
          for(int j = 0; j < kernel_height; j++)
          {
              for(int k = 0; k < kernel_width; k++)
              {
                  kernel[i * kernel_height * kernel_width + j * kernel_width + k] = kernel_tmp[l][i][j][k];     
              }
          }    
        }
       float* actual_result = convolve_FFT(input_layer, kernel, pad, stride, batch_size, il_dim, kernel_dim);
        for(int ll = 0; ll < batch_size; ll++)  
        {
          for(int ii = 0; ii < out_H; ii++)
          {
              for(int jj = 0; jj < out_W; jj++)
              {
                    //printf("%f ",round(actual_result[ll*out_H * out_W + ii*out_W+jj]));
                    final_output[ll*out_size*out_H*out_W + l*out_H*out_W + ii * out_W + jj] = round(actual_result[ll*out_H * out_W + ii*out_W+jj]);
              }
              //printf("\n");
          }
          //printf("\n\n");
        }
       
   }

    for(int l = 0; l < batch_size; l++)
    {
      for(int i = 0; i < out_size; i++)
      {
        for(int j = 0; j < out_H; j++)
        {
            for(int k = 0; k < out_W; k++)
            {
              printf("%f ",  final_output[l * out_size* out_H * out_W + i * out_H * out_W + j * out_W + k]);
            }
          printf("\n");
        }    
        printf("\n");
      }
      printf("\n\n");
    }

   return 0;
}