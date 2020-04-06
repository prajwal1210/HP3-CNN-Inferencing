#include <"fft.cu">

int main()
{
   int channel = 3;
  int height = 5;
  int width = 5;
  int kernel_height = 3;
  int kernel_width = 3;
  int batch_size = 3;
  int pad = 0;
  int stride = 2;
  int out_size = 2;   
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
    
    float* input_layer = (float *)malloc(batch_size* channel * height* width * sizeof(float));
    float* kernel = (float *)malloc(out_size* channel * kernel_height* kernel_width * sizeof(float));
   int out_H = ((height - kernel_height + 2 * pad)/stride) + 1; 
  int out_W = ((width - kernel_width + 2 * pad)/stride) + 1;
 
   for(int l = 0; l < batch_size; l++)
    {
      for(int i = 0; i < channel; i++)
      {
        for(int j = 0; j < height; j++)
        {
            for(int k = 0; k < width; k++)
            {
                input_layer[l * channel * height * width + i * height * width + j * width + k] = input_layer_tmp[l][i][j][k];  
            }
        }    
      }
    }
 
    for(int l = 0; l < out_size ; l++)
   {
        for(int i = 0; i < channel; i++)
        {
          for(int j = 0; j < kernel_height; j++)
          {
              for(int k = 0; k < kernel_width; k++)
              {
                  kernel[l * channel * kernel_height * kernel_width + i * kernel_height * kernel_width + j * kernel_width + k] = kernel_tmp[l][i][j][k];     
              }
          }    
        }
   }
 
   float* final_output =  forward(out_size, channel, kernel_height, kernel_width, pad, stride, kernel, batch_size, height, width, input_layer);
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