
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

