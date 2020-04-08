
template<typename T> __device__ inline void __swap(T &mem1, T&mem2)
{ T temp = mem1; mem1 = mem2, mem2 = temp; }

__global__ void solve(float *mem, size_t n) {
    int i = threadIdx.x, j = threadIdx.y;

    if( j % 2 == 0 and j+1 < n )
        __swap(mem[i*n+j], mem[i*n+j+1]);

    __syncthreads();

    if( j > i )
        mem[i*n+j] = mem[j*n+i];

}

// forward(int out_channels, int input_channels, int kernel_height, int kernel_width, int padding, int stride, float* kernel_weights, int batchsize_of_data, int input_height, int input_width, float* input)

// Tiling kernel

// grid : (bs, p, q)
// block : (ch, 1, 1)

// Here, p and q are the tiles dimension for the image.
// So, if the image is of dimension (h, w) after tiling it would become of dimension (p, q) of tiles of dimension (4, 4)

// Here, we haven't considered multiple output channels yet. This dimension can be included in the grid like (och x bs, p, q) or in the host function itself we can call a loop on the current version over multiple output channels.

// kernel height and width are fixed to (3, 3)

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