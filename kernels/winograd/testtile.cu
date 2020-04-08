%%cuda --name helloCUDA.cu
#include <iostream>
#include <random>
#include <algorithm>

#define LOOP(x) for(int t##x = 0; t##x < x; t##x++)

using namespace std;

__global__ void tile(float *devin, float *devout, int h, int w)
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

    int offset = (tbs*ch + tch)*h*w;

    // float *t = thrtile;
 
    for(int th = 2*tp, i = 0; i < 4; th++, i++)
    {
        for(int tw = 2*tq, j = 0; j < 4; tw++, j++)
        {
            thrtile[i][j] = devin[offset + th*w + tw];
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
}

void gpu_error(cudaError_t const &code) 
{
    if(code != cudaSuccess)
    {
        cerr << "GPUError: Code " << code << " : " << cudaGetErrorString(code) << endl;
        exit( EXIT_FAILURE );
    }
}

void tilehost(int och, int ch, int bs, int h, int w, float *&in, int &p, int &q, int &outsize, float *&out)
{
    // int p, q;
    p = max((h-2)/2, 0);
    q = max((w-2)/2, 0);
    
    float *devin, *devout;
    devin = devout = nullptr;
    int insize = bs * ch * h * w * sizeof(float);
    outsize = bs * p * q * ch * 4 * 4 * sizeof(float);

    gpu_error(cudaMalloc((void **) & devin, insize));
    gpu_error(cudaMalloc((void **) & devout, outsize));
    
    gpu_error(cudaMemcpy(devin, in, insize, cudaMemcpyHostToDevice));

    // call the kernel function for tiling
    
    dim3 grid(bs, p, q);  // 3-D
    dim3 block(ch, 1, 1); // 1-D

    tile<<<grid, block>>>(devin, devout, h, w);

    // copy from device to host to out.

    delete in;
    out = new float[outsize];

    gpu_error(cudaMemcpy(out, devout, outsize, cudaMemcpyDeviceToHost));
    
}

int main(void) 
{
    auto engine = default_random_engine(time(nullptr));
    auto rng = uniform_real_distribution<float>();

    int bs, ch, h, w, p, q;
    
    bs = 3;
    ch = 2;
    h = 9;
    w = 9;
    
    int insize = bs * ch * h * w * sizeof(float);
    int outsize;
 
    float *in = new float[insize];
    float *t = in;
    float *out;
 
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

    cout<<"\nTiling\n";

    tilehost(1, ch, bs, h, w, in, p, q, outsize, out);
    
    cout<<"Tiling finished\n\n";

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

    return 0;
}