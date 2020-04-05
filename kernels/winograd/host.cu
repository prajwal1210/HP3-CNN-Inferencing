#include "header.h"
#define LOOP(x) for(int t##x = 0; t##x < x; t##x++)

using namespace std;

void gpu_error(cudaError_t const &code) 
{
    cerr << "GPUError: Code " << code << " : " << cudaGetErrorString(code) << endl;
    exit( EXIT_FAILURE );
}

// forward(int out_channels, int input_channels, int kernel_height, int kernel_width, int padding, int stride, float* kernel_weights, int batchsize_of_data, int input_height, int input_width, float* input)

void rearrange(int ch, int bs, int h, int w, float *& in)
{
    // ch   : input_channels
    // bs   : batchsize_of_data
    // h    : input_height
    // w    : input_width
    // in   : reference to float pointer input
    // This function transforms the input from bs x ch x h x w to ch x h x w x bs

    float *newin = new float[ch * h * w * bs];
    float *newiter = newin;
    int ch_h_w = ch*h*w, h_w = h*w; 
    LOOP(ch)
    {
        LOOP(h)
        {
            LOOP(w)
            {
                LOOP(bs)
                {
                    *(newiter++) = in[tbs * ch_h_w + tch * h_w + th * w + tw];
                }
            }
        }
    }

    delete in;

    in = newin;

}

int main(void) {
    auto engine = default_random_engine(time(nullptr));
    auto rng = uniform_real_distribution<float>();

    size_t n; cin >> n;
    if( n > N_MAX ) {
        cerr << "Size " << n << " too large." << endl;
        return EXIT_FAILURE;
    }

    size_t mat_size = n * n;
    float P[mat_size], Q[mat_size], *d_P = nullptr;
    for(size_t i=0; i<mat_size; i++) Q[i] = P[i] = rng(engine);

    cudaError_t code = cudaSuccess;
    if( (code = cudaMalloc((void**)&d_P, sizeof P)) != cudaSuccess ) gpu_error(code);
    if( (code = cudaMemcpy(d_P, P, sizeof P, cudaMemcpyHostToDevice)) != cudaSuccess ) gpu_error(code);

    solve<<< 1,dim3(n,n) >>> (d_P, n);

    if( (code = cudaMemcpy(Q, d_P, sizeof P, cudaMemcpyDeviceToHost)) != cudaSuccess ) gpu_error(code);
    if( (code = cudaFree(d_P)) != cudaSuccess ) gpu_error(code);
    d_P = nullptr;

    if( not check(n, P, Q) ) {
        cerr << "Test failed." << endl;
        return EXIT_FAILURE;
    }

    cout << "Test passed." << endl;
    return EXIT_SUCCESS;
}
