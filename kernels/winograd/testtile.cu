%%cuda --name helloCUDA.cu
#include <iostream>
#include <random>
#include <algorithm>

#define LOOP(x) for(int t##x = 0; t##x < x; t##x++)

using namespace std;

__global__ void tile(float *devin, float *devout, float *devsum, int h, int w)
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

    // copy thrtile to devout for testing

    int offset2 = (((tbs*p + tp)*q + tq)*ch + tch)*16;

    for(int i = 0; i < 4; i++)
    {
        for(int j = 0; j < 4; j++)
        {
            devout[offset2 + i*4 + j] = thrtile[i][j];
        }
    }

    // sum along the channels, using log n summing

    // int k = ch, j = tch;

    int offset3 = ((tbs*p + tp)*q + tq)*ch*16;

    for(int s = 1; s < ch; s *= 2)
    {
        if(tch % (2*s) == 0 && tch+s < ch)
        {
            for(int i = 0; i < 4; i++)
            {
                for(int j = 0; j < 4; j++)
                {
                    devout[offset3 + tch*16 + i*4 + j] += devout[offset3 + (tch+s)*16 + i*4 + j];
                }
            }
        }
        __syncthreads();
    }

    if(tch/*%ch*/ == 0) // can do with tch == 0
    {
        int offset = ((tbs*p + tp)*q + tq)*16;
        for(int i = 0; i < 4; i++)
        {
            for(int j = 0; j < 4; j++)
            {
                devsum[offset + i*4 + j] = devout[offset3 + /*tch*16*/ +i*4 + j];
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

void tilehost(int och, int ch, int bs, int h, int w, float *&in, int &p, int &q, int &outsize, float *&out, int &sumsize, float *&sum)
{
    // int p, q;
    p = max((h-2)/2, 0);
    q = max((w-2)/2, 0);
    
    float *devin, *devout, *devsum;
    devin = devout = devsum = nullptr;
    int insize = bs * ch * h * w * sizeof(float);
    outsize = bs * p * q * ch * 4 * 4 * sizeof(float);
    sumsize = bs * p * q * 4 * 4 * sizeof(float);

    gpu_error(cudaMalloc((void **) & devin, insize));
    gpu_error(cudaMalloc((void **) & devout, outsize));
    gpu_error(cudaMalloc((void **) & devsum, sumsize));
    
    gpu_error(cudaMemcpy(devin, in, insize, cudaMemcpyHostToDevice));

    // call the kernel function for tiling
    
    dim3 grid(bs, p, q);  // 3-D
    dim3 block(ch, 1, 1); // 1-D

    tile<<<grid, block>>>(devin, devout, devsum, h, w);

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

    tilehost(1, ch, bs, h, w, in, p, q, outsize, out, sumsize, sum);
    
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