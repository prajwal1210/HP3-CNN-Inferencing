%%cuda --name helloCUDA.cu
#include <iostream>
#include <random>
#include <algorithm>

#define LOOP(x) for(int t##x = 0; t##x < x; t##x++)

using namespace std;

__global__ precompute(int out_channels, int input_channels, float* kernel_weights, float *U)
{
    int x = threadIdx.x;
    int y = blockDim.x;;
    int bid = blockIdx.x;
    int offset = bid*y + x;
    int m = 2, n = 3;
    
    float g[4][3] = {
        {1, 0, 0},
        {0.5, 0.5, 0.5},
        {0.5, -0.5, 0.5},
        {0, 0, 1}
    };
    
    float g_t[3][4] ={
        {1, 0.5, 0.5, 0},
        {0, 0.5, -0.5, 0},
        {0, 0.5, 0.5, 1}
    };

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

__global__ void uv(int tch, int out_channels, float *fin, float *U, float V[4][4])
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

__global__ void tile(float *devin, float *devout, float *devsum, float *U, int h, int w, int och)
{
    float thrtile[4][4];
    
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

    int offset1 = (tbs*ch + tch)*h*w;

    // float *t = thrtile;
 
    for(int th = 2*tp, i = 0; i < 4; th++, i++)
    {
        for(int tw = 2*tq, j = 0; j < 4; tw++, j++)
        {
            thrtile[i][j] = devin[offset1 + th*w + tw];
        }
    }
    float B[4][4] = {
        {1,0,0,0},
        {0,1,-1,1},
        {-1,1,1,0},
        {0,0,0,-1}
    };

    float B_t[4][4] = {
        {1,0,-1,0},
        {0,1,1,0},
        {0,-1,1,0},
        {0,1,0,-1}
    };
    //Calculation of V
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
/

    float *fin = (float *)malloc(och*4*4);
    uv<<<1,och>>>(tch, och, fin, U, V); 

    // copy thrtile to devout for testing

    int offset2 = (((tbs*p + tp)*q + tq)*ch + tch)*16;
    LOOP(och)
    {
        for(int i = 0; i < 4; i++)
        {
            for(int j = 0; j < 4; j++)
            {
                devout[(((((tbs*och+toch)*p+tp)*q+tq)*ch+tch)*4 + i)*4 + j] = fin[(toch*4+i)*4+j];
            }
        }
    }

    // sum along the channels, using log n summing

    // int k = ch, j = tch;

    int offset3 = ((tbs*p + tp)*q + tq)*ch*16;

    for(int s = 1; s < ch; s *= 2)
    {
        if(tch % (2*s) == 0 && tch+s < ch)
        {
            LOOP(och)
            {
                for(int i = 0; i < 4; i++)
                {
                    for(int j = 0; j < 4; j++)
                    {
                        devout[(((((tbs*och+toch)*p+tp)*q+tq)*ch+tch)*4 + i)*4 + j] += devout[(((((tbs*och+toch)*p+tp)*q+tq)*ch+(tch+s))*4 + i)*4 + j];
                    }
                }
            }
        }
        __syncthreads();
    }

    if(tch/*%ch*/ == 0) // can do with tch == 0
    {
        int offset = ((tbs*p + tp)*q + tq)*16;
        LOOP(och)
        {    
            for(int i = 0; i < 4; i++)
            {
                for(int j = 0; j < 4; j++)
                {
                    devsum[((((tbs*och+toch)*p+tp)*q+tq)*4 + i)*4 + j] = devout[(((((tbs*och+toch)*p+tp)*q+tq)*ch)*4 + i)*4 + j];
                }
            }
        }
    }

}

void gpu_error(cudaError_t const &code) 
{
    if(code != cudaSuccess)
    {
        cerr << "GPUError: Code " << code << " : " << cudaGetErrorString(code) << endl;
        exit( EXIT_FAILURE );
    }
}

void tilehost(int och, int ch, int bs, int h, int w, float *&in, int &p, int &q, int &outsize, float *&out, int &sumsize, float *&sum, float *kernel_weights)
{
    // int p, q;
    p = max((h-2)/2, 0);
    q = max((w-2)/2, 0);
    
    float *devin, *devout, *devsum;
    devin = devout = devsum = nullptr;
    int insize = bs * ch * h * w * sizeof(float);
    outsize = bs * och * p * q * ch * 4 * 4 * sizeof(float);
    sumsize = bs * och * p * q * 4 * 4 * sizeof(float);

    gpu_error(cudaMalloc((void **) & devin, insize));
    gpu_error(cudaMalloc((void **) & devout, outsize));
    gpu_error(cudaMalloc((void **) & devsum, sumsize));
    
    gpu_error(cudaMemcpy(devin, in, insize, cudaMemcpyHostToDevice));

    // call the kernel function for tiling
    
    float *U = (float *)malloc(och*ch*4*4*sizeof(float));
    precompute<<<och, ch>>>(och, ch, kernel_weights, U);

    dim3 grid(bs, p, q);  // 3-D
    dim3 block(ch, 1, 1); // 1-D
    tile<<<grid, block>>>(devin, devout, devsum, U, h, w, och);

    // copy from device to host to out.

    delete in;
    out = new float[outsize/sizeof(float)];
    sum = new float[sumsize/sizeof(float)];

    gpu_error(cudaMemcpy(out, devout, outsize, cudaMemcpyDeviceToHost));
    gpu_error(cudaMemcpy(sum, devsum, sumsize, cudaMemcpyDeviceToHost));

    gpu_error(cudaFree(devin));
    gpu_error(cudaFree(devout));
    gpu_error(cudaFree(devsum));
    
}

int main(void) 
{
    auto engine = default_random_engine(time(nullptr));
    auto rng = uniform_real_distribution<float>();

    int bs, ch, h, w, p, q;
    
    bs = 1;
    ch = 2;
    h = 9;
    w = 9;
    
    int insize = bs * ch * h * w * sizeof(float);
    int outsize, sumsize;
 
    float *in = new float[insize/sizeof(float)];
    float *t = in;
    float *out, *sum;
 
    LOOP(bs)
    {
        LOOP(ch)
        {
            LOOP(h)
            {
                LOOP(w)
                {
                    *(t++) = rng(engine);
                }
            }
        }
    }
 
    LOOP(bs)
    {
        cout<<"{ ";
        LOOP(ch)
        {
            cout<<"{ ";
            LOOP(h)
            {
                cout<<"{ ";
                LOOP(w)
                {
                    cout<<in[((tbs*ch+tch)*h+th)*w+tw]<<" ";
                }
                cout<<"}\n";
            }
            cout<<"}\n";
        }
        cout<<"}\n";
    }

    cout<<"\nTiling and Summing\n";

    tilehost(1, ch, bs, h, w, in, p, q, outsize, out, sumsize, sum, kernel_weights);
    
    cout<<"\nTiling finished\n\n";

    /*
    
    LOOP(bs)
    {
        cout<<"{ ";
        LOOP(p)
        {
            cout<<"{ ";
            LOOP(q)
            {
                cout<<"{ ";
                LOOP(ch)
                {
                    cout<<"{ ";
                    for(int i = 0; i < 4; i++)
                    {
                        for(int j = 0; j < 4; j++)
                        {
                            cout<<out[((((tbs*p+tp)*q+tq)*ch+tch)*4+i)*4+j]<<",";
                        }
                        cout<<";\n";
                    }
                    cout<<"}\n";
                }
                cout<<"}\n";
            }
            cout<<"}\n";
        }
        cout<<"}\n";
    }
 
    */

    cout<<"\nSumming finished\n\n";

    LOOP(bs)
    {
        cout<<"{ ";
        LOOP(p)
        {
            cout<<"{ ";
            LOOP(q)
            {
                cout<<"{ ";
                for(int i = 0; i < 4; i++)
                {
                    for(int j = 0; j < 4; j++)
                    {
                        cout<<sum[(((tbs*p+tp)*q+tq)*4+i)*4+j]<<",";
                    }
                    cout<<";\n";
                }
                cout<<"}\n";
            }
            cout<<"}\n";
        }
        cout<<"}\n";
    }

    return 0;
}