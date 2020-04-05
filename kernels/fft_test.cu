int main(){
    int channel = 3;
    int height = 5;
    int width = 5;
    int kernel_height = 3;
    int kernel_width = 3;
    float input_layer_tmp[channel][height][width]  = {{{0,0,1,0,2},
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
    /*float input_layer_tmp[channel][height][width]  = {{{0,0,1,0,2},
      {1,0,2,0,1},
      {1,0,2,2,0},
      {2,0,0,2,0},
      {2,1,2,2,0}}};*/
    
    float* input_layer = (float *)malloc(channel * height* width * sizeof(float));
    float* kernel = (float *)malloc(channel * kernel_height* kernel_width * sizeof(float));  

    float kernel_tmp[channel][kernel_height][kernel_width] = {{{-1,0,1},
      {0,0,1},
      {1,-1,1}},{{-1,0,1},
      {1,-1,1},
      {0,1,0}},{{-1,1,1},
      {1,1,0},
      {0,-1,0}}};
    /*float kernel_tmp[channel][kernel_height][kernel_width] = {{{-1,0,1},
      {0,0,1},
      {1,-1,1}}};*/

    for(int i = 0; i < channel; i++)
    {
      for(int j = 0; j < height; j++)
      {
          for(int k = 0; k < width; k++)
          {
              input_layer[i + j * channel+ k * channel * height] = input_layer_tmp[i][j][k];  
             // printf("%f ", input_layer_tmp[i][j][k]);
          }
         //printf("\n");
      }    
      //printf("\n");
    }

    for(int i = 0; i < channel; i++)
    {
      for(int j = 0; j < kernel_height; j++)
      {
          for(int k = 0; k < kernel_width; k++)
          {
              kernel[i + j * channel + k * channel * kernel_height] = kernel_tmp[i][j][k];  
          }
      }    
    }
      
    int pad = 0;
    int stride = 2;
    int batch_size = 1;
    int il_dim[3] = {height, width, channel}; int kernel_dim[3] = {kernel_height, kernel_width, channel};
    float* actual_result;
    //float expected_result[height][width] = {{2,3,3},{3,7,3},{8,10,-3}};
    int out_H = ((height - kernel_height + 2 * pad)/stride) + 1; 
    int out_W = ((width - kernel_width + 2 * pad)/stride) + 1; 
    
    actual_result = convolve_FFT(input_layer, kernel, pad, stride, batch_size, il_dim, kernel_dim);
    for(int i = 0; i < out_H; i++)
      {
          for(int j = 0; j < out_W; j++)
          {
              //if(abs(actual_result[i + j*height] - expected_result[i][j]) > 0.001 ){
                printf("%f ",round(actual_result[i+j*out_H]));
              //}
          }
          printf("\n");
      }
      //printf("Test Passed.\n");
    return 0;
}