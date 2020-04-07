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
    
   float* input_layer_cuda = NULL; cudaMalloc((void **)&input_layer_cuda, batch_size* channel * height* width * sizeof(float));
   float* kernel_cuda = NULL; cudaMalloc((void **)&kernel_cuda, out_size* channel * kernel_height* kernel_width * sizeof(float));
   cudaMemcpy(input_layer_cuda, input_layer_tmp , batch_size * channel * height* width * sizeof(float) ,cudaMemcpyHostToDevice);
   cudaMemcpy(kernel_cuda, kernel_tmp , out_size *channel * kernel_height* kernel_width * sizeof(float) ,cudaMemcpyHostToDevice);
   
 
   float* final_output =  forward(out_size, channel, kernel_height, kernel_width, pad, stride, kernel_cuda, batch_size, height, width, input_layer_cuda);
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