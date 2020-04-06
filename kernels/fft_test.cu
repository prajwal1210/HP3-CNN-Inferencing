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

    float kernel_tmp[channel][kernel_height][kernel_width] = 
        {
            {{-1,0,1}, {0,0,1}, {1,-1,1}},
            {{-1,0,1}, {1,-1,1}, {0,1,0}},
            {{-1,1,1}, {1,1,0}, {0,-1,0}}
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

    for(int i = 0; i < channel; i++)
    {
      for(int j = 0; j < kernel_height; j++)
      {
          for(int k = 0; k < kernel_width; k++)
          {
              kernel[i * kernel_height * kernel_width + j * kernel_width + k] = kernel_tmp[i][j][k]; 
              //printf("%f ",  kernel[i * kernel_height * kernel_width + j * kernel_width + k]); 
          }//printf("\n");
      }    
      //printf("\n");
    }
      
    int il_dim[3] = {height, width, channel}; int kernel_dim[3] = {kernel_height, kernel_width, channel};
    float* actual_result;
    //float expected_result[height][width] = {{2,3,3},{3,7,3},{8,10,-3}};
    int out_H = ((height - kernel_height + 2 * pad)/stride) + 1; 
    int out_W = ((width - kernel_width + 2 * pad)/stride) + 1; 
    
    actual_result = convolve_FFT(input_layer, kernel, pad, stride, batch_size, il_dim, kernel_dim);
    for(int l = 0; l < batch_size; l++)  
    {
      for(int i = 0; i < out_H; i++)
      {
          for(int j = 0; j < out_W; j++)
          {
              //if(abs(actual_result[i * width + j] - expected_result[i][j]) > 0.001 ){
                printf("%f ",round(actual_result[i*out_W+j]));
              //}
          }
          printf("\n");
      }
       printf("\n\n");
    }
      //printf("Test Passed.\n");
    return 0;
}