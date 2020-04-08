#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ precompute(int out_channels, int input_channels, float* kernel_weights, float *U)
{
    int x = threadIdx.x;
    int y = blockDim.x;;
    int bid = blockIdx.x;
    int offset = bid*y + x;
    int m = 2, n = 3;
    float g[4][3], g_t[3][4];
    float *temp = (float *)malloc(out_channels*input_channels*3*4*sizeof(float));
    for(int i = 0; i <3; ++i)
    {
        for(int j = 0; j <4; ++j)
        {
            temp[offset*3*4+i*4+j] = 0;
            for(int k = 0; k <3; ++k)
            {
                temp[offset*3*4+i*4+j] += kernel_weights[offset*3*3+i*3+k] * g_t[k][j];
            }
        }
    }

    for(int i = 0; i <4; ++i)
    {
        for(int j = 0; j <4; ++j)
        {
            U[offset*4*4+i*4+j] = 0;
            for(int k = 0; k <3; ++k)
            {
                U[offset*4*4+i*4+j] += g[i][k] * temp[offset*3*4+k*4+j];
            }
        }
    }
}

__global__ void uv(int tch, int out_channels, float *fin, float V[4][4])
{
    int x = threadIdx.x;
    int offset = x*out_channels+tch;

    for(int i = 0; i <4; ++i)
    {
        for(int j = 0; j <4; ++j)
        {
            fin[x*out_channels*4*4 + i*4 + j] = U[offset*4*4+i*4+j]*V[i][j];
            
        }
    }

}

__global__ void tile(float *devin, float *devout, int h, int w)
{
    float thrtile[4][4];
    float V[4][4];
    
    int /*bs,*/ p, q, ch;
    // bs = gridDim.x;
    p = gridDim.y;
    q = gridDim.z;
    ch = blockDim.x;
    
    int tbs, tp, tq, tch;
    tbs = blockIdx.x;
    tp = blockIdx.y;
    tq = blockIdx.z;
    tch = threadIdx.x;

    // copy the tiles to thrtile

    int offset = (tbs*ch + tch)*h*w;

    for(int th = 2*tp, i = 0; i < 4; th++, i++)
    {
        for(int tw = 2*tq, j = 0; j < 4; tw++, j++)
        {
            thrtile[i][j] = devin[offset + th*w + tw];
        }
    }

    // Calculation of V
    for(int i = 0; i <4; ++i)
    {
        for(int j = 0; j <4; ++j)
        {
            temp[i][j] = 0;
            for(int k = 0; k <4; ++k)
            {
                temp[i][j] += thrtile[i][k] * B[k][j];
            }
     
        }
    }

    for(int i = 0; i <4; ++i)
    {
        for(int j = 0; j <4; ++j)
        {
            V[i][j] = 0;
            for(int k = 0; k <4; ++k)
            {
                V[i][j] += B_t[i][k] * temp[k][j];
            }
        }
    }

    float *fin = (float *)malloc(out_channels*4*4);
    uv<<<1,out_channels>>>(tch, out_channels, fin, V); 

    // copy thrtile to devout for testing

    // int offset2 = (((tbs*p + tp)*q + tq)*ch + tch)*16;

    // for(int i = 0; i < 4; i++)
    // {
    //     for(int j = 0; j < 4; j++)
    //     {
    //         devout[offset2 + i*4 + j] = thrtile[i][j];
    //     }
    // }
}

// __global__ calc()
// {
//     int m = 2, n = 3;
//     int B[4][4], B_t[4][4];
//     float *temp = (float *)malloc(input_channels*4*4*sizeof(float));
//     float *V = (float *)malloc(input_channels*4*4*sizeof(float));
//     for(int i = 0; i <4; ++i)
//     {
//         for(int j = 0; j <4; ++j)
//         {
//             temp[x*n*n+i*n+j] = 0;
//             for(int k = 0; k <4; ++k)
//             {
//                 temp[x*n*n+i*n+j] += d[i*n+k] * B[k][j];
//             }
//         }
//     }

//     for(int i = 0; i <4; ++i)
//     {
//         for(int j = 0; j <4; ++j)
//         {
//             U[x*n*n+i*n+j] = 0;
//             for(int k = 0; k <n; ++k)
//             {
//                 V[x*n*n+i*n+j] += B_t[i][k] * temp[x*n*n+k*n+j];
//             }
//         }
//     }


// }

void forward(int out_channels, int input_channels, int kernel_height, int kernel_width, int padding, int stride, float* kernel_weights
		int batchsize_of_data, int input_height, int input_width, float* input)
{
    int m = 2, n = 3;
    float *U = (float *)malloc(out_channels*input_channels*4*4*sizeof(float));
	precompute<<<out_channels, input_channels>>>(out_channels, input_channels, kernel_weights, U);
    
	calc<<<>>>(U, )

}



int main()
{

}