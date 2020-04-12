#include "wingheader.h"

#define LOOP(x) for(int t##x = 0; t##x < x; t##x++)
#define cudaSafeCall(call)  \
        do {\
            cudaError_t err = call;\
            if (cudaSuccess != err) \
            {\
                std::cerr << "CUDA error in " << __FILE__ << "(" << __LINE__ << "): " \
                    << cudaGetErrorString(err);\
                exit(EXIT_FAILURE);\
            }\
        } while(0)


void gpu_error(cudaError_t const &code) {
    if(code != cudaSuccess)
    {
        std::cerr << "GPUError: Code " << code << " : " << cudaGetErrorString(code) << std::endl;
        exit( EXIT_FAILURE );
    }
}

__global__ void precompute(int och, int ch, float* kernel_weights, float *U)
{
    int x = threadIdx.x;
    int bid = blockIdx.x;
    int offset = bid*ch + x;
   
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
    float temp[3][4];// = (float *)malloc(3*4*sizeof(float));
    for(int i = 0; i <3; ++i)
    {
        for(int j = 0; j <4; ++j)
        {
            temp[i][j] = 0;
            for(int k = 0; k <3; ++k)
            {
                temp[i][j] += kernel_weights[offset*3*3+i*3+k] * g_t[k][j];
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
                U[offset*4*4+i*4+j] += g[i][k] * temp[k][j];
            }
        }
    }
    // free(temp);
}
__device__ void uv(int tbs, int tp, int tq, int p, int q, int och, int tch, int ch, float *devfin, float *U,  float V[4][4])
{
    int x = 0;//threadIdx.x;
    for(int i = 0; i <4; ++i)
        for(int j = 0; j <4; ++j)
            devfin[(((((tbs*p+tp)*q+tq)*ch+tch)*och+x)*4+i)*4+j] = U[((x*ch+tch)*4+i)*4+j]*V[i][j];            
}
__device__ void amul(int tbs, int tp, int tq, int bs, int och, int p, int q, float *devsum, float *devY)
{
    float A_t[2][4] = {
        {1, 1, 1, 0},
        {0, 1, -1,-1}
    };
    float A[4][2] = {
        {1,0},
        {1,1},
        {1,-1},
        {0,-1}
    };
    int x = 0; //threadIdx.x;
    float temp[2][4];// = (float *)malloc(2*4*sizeof(float));
    for(int i = 0; i <2; ++i)
    {
        for(int j = 0; j <4; ++j)
        {
            temp[i][j] = 0;
            for(int k = 0; k <4; ++k)
            {
                temp[i][j] += A_t[i][k] * devsum[((((tbs*och+x)*p+tp)*q+tq)*4+k)*4+j];
            }
        }
    }
    for(int i = 0; i <2; ++i)
    {
        for(int j = 0; j <2; ++j)
        {
            devY[((((tbs*och+x)*p+tp)*q+tq)*2+i)*2+j] = 0;
            for(int k = 0; k <4; ++k)
            {
                devY[((((tbs*och+x)*p+tp)*q+tq)*2+i)*2+j] += temp[i][k] * A[k][j];
            }
        }
    }
    // free(temp);
}

__global__ void paddev(float *devin, float *devinnopad, int h, int w, int pad)
{
    int newh = gridDim.y;
    int neww = gridDim.z;
    int tbsch = blockIdx.x;
    int tnewh = blockIdx.y;
    int tneww = blockIdx.z;
    int newhw = newh*neww;
    int hw = h*w;
    int th = tnewh-pad;
    int tw = tneww-pad;
    
    if(th >= 0 && th < h && tw >= 0 && tw < w)
        devin[tbsch*newhw + tnewh*neww + tneww] = devinnopad[tbsch*hw + th*w + tw];
    else
        devin[tbsch*newhw + tnewh*neww + tneww] = 0;
    
}

__global__ void cutpad(float  *devY, float *devcutY, int oph,int opw)
{
    int p = gridDim.y;
    int q = gridDim.z;
    int tbsch = blockIdx.x;
    int tp = blockIdx.y;
    int tq = blockIdx.z;
    //int newhw = newh*neww;
    //int pq4 = p*q*4;
    int ophopw = oph*opw;
    for(int i = 0; i < 2; i++)
    {
        for(int j = 0; j < 2; j++)
        {
            if(tp*2 + i < oph && tq*2 + j < opw)
                devcutY[tbsch*ophopw + (tp*2+i)*opw + (tq*2+j)] = devY[(((tbsch*p + tp)*q +tq)*2 + i)*2  + j];
        }
    }
}
    
__global__ void tile(int bs, int p, int q, int ch, float *devin, float *devout, float *devsum, float *devY, float *devU, int h, int w, int och, float *devfin)
{
    float thrtile[4][4];    
    int tbs, tp, tq, tch;
    tbs = blockIdx.x;
    tp = blockIdx.y;
    tq = blockIdx.z;
    tch = threadIdx.x;
    // copy the tiles to thrtile
    int offset1 = (tbs*ch + tch)*h*w;
    for(int th = 2*tp, i = 0; i < 4; th++, i++)
        for(int tw = 2*tq, j = 0; j < 4; tw++, j++)
            thrtile[i][j] = devin[offset1 + th*w + tw];

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
    float temp[4][4];
    float V[4][4];// = (float *)  malloc(16*sizeof(float));
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
    //float *fin = (float *)malloc(och*4*4*sizeof(float));
    uv(tbs, tp, tq, p, q, och, tch, ch, devfin, devU, V); 
    //cudaDeviceSynchronize();
    //free(V);

    for(int toch = 0; toch<och; toch++)
        for(int i = 0; i < 4; i++)
            for(int j = 0; j < 4; j++)
               devout[(((((tbs*och+toch)*p+tp)*q+tq)*ch+tch)*4 + i)*4 + j] = devfin[(((((tbs*p+tp)*q+tq)*ch+tch)*och+toch)*4+i)*4+j];

    // sum along the channels, using log n summing
    //free(fin);
    for(int s = 1; s < ch; s *= 2)
    {
        if(tch % (2*s) == 0 && tch+s < ch)
        {
            LOOP(och)
                for(int i = 0; i < 4; i++)
                    for(int j = 0; j < 4; j++)
                        devout[(((((tbs*och+toch)*p+tp)*q+tq)*ch+tch)*4 + i)*4 + j] += devout[(((((tbs*och+toch)*p+tp)*q+tq)*ch+(tch+s))*4 + i)*4 + j];
        }
        __syncthreads();
    }
    if(tch == 0) 
    {
        LOOP(och)
            for(int i = 0; i < 4; i++)
                for(int j = 0; j < 4; j++)
                    devsum[((((tbs*och+toch)*p+tp)*q+tq)*4 + i)*4 + j] = devout[(((((tbs*och+toch)*p+tp)*q+tq)*ch)*4 + i)*4 + j];
    }
  if(tch == 0)
  {
      amul(tbs, tp, tq, bs, och, p, q, devsum, devY);
      //cudaDeviceSynchronize();
  }
}
void tilehost(int p, int q, int och, int ch, int bs, int h, int w, int pad, float *devin, int oph, int opw, 
    float *devU, float *cutY, float *devout, float *devsum, float *devfin, float *devY, float *devcutY, float& conv_time, float& overhead_time) {
    
    float milliseconds = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    dim3 grid(bs, p, q);  // 3-D
    dim3 block(ch, 1, 1); // 1-D
    
    cudaEventRecord(start);
    // call the kernel function for tiling
    tile<<<grid, block>>>(bs, p, q, ch, devin, devout, devsum, devY, devU, h, w, och, devfin);
    //cudaSafeCall(cudaGetLastError());

    
    size_t cutsize = bs*och*oph*opw*sizeof(float);
    dim3 cutgrid(bs*och, p, q);
    dim3 cutblock(1,1,1);
    
    cutpad<<<cutgrid, cutblock>>> (devY, devcutY, oph, opw);   
    cudaEventRecord(stop);
    // copy from device to host.
    //delete in;
    // cutY = (float *)malloc(cutsize);

    cudaSafeCall(cudaMemcpy(cutY, devcutY, cutsize, cudaMemcpyDeviceToHost));
    
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    conv_time += milliseconds;
    // return cutY;
}

__global__ void ucopy(float *devtempU, float *devU, int toch, int n1, int n2, int ch)
{
    int tch = threadIdx.x;
    LOOP(n1)
        LOOP(n2)
             devtempU[((tch*n1+tn1)*n2+tn2)] = devU[(((toch*ch+tch)*n1+tn1)*n2+tn2)];
}

float * WING::forward(int och, int ch, int bs, int h, int w, int pad, float *in, int &oph, int &opw, float *kwt, float& conv_time, float& overhead_time) {
    conv_time = 0;
    overhead_time = 0;
    float milliseconds = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    float *devin, *devinnopad, *cutY, *devkwt, *devU, *devtempU;
    size_t insize = bs * ch * h * w * sizeof(float);
    int newh, neww;
 
    gpu_error(cudaMalloc((void **) & devinnopad, insize));
    gpu_error(cudaMemcpy(devinnopad, in, insize, cudaMemcpyHostToDevice));

    newh = h + 2*pad;
    neww = w + 2*pad;
    oph = newh-2;
    opw = neww-2;
    if(newh%2)
        newh++;
    if(neww%2)
        neww++;
    if(newh < 4)
        newh = 4;
    if(neww < 4)
        neww = 4;

    insize = bs * ch * newh * neww * sizeof(float);
    gpu_error(cudaMalloc((void **) & devin, insize));

    // call padding
    dim3 padgrid(bs*ch, newh, neww);
    dim3 padblock(1, 1, 1);

    cudaEventRecord(start);
    paddev<<<padgrid,padblock>>>(devin, devinnopad, h, w, pad);
    cudaEventRecord(stop);

    gpu_error(cudaFree(devinnopad));

    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    overhead_time += milliseconds;

    h = newh;
    w = neww;
    int p = max((h-2)/2, 0);
    int q = max((w-2)/2, 0);

    size_t kwtsize = och*ch*3*3*sizeof(float);    
    size_t usize = och*ch*4*4*sizeof(float);
    gpu_error(cudaMalloc((void **) & devkwt, kwtsize));
    gpu_error(cudaMalloc((void **) & devU, usize));
    gpu_error(cudaMemcpy(devkwt, kwt, kwtsize, cudaMemcpyHostToDevice));
    
    cudaEventRecord(start);
    precompute<<<och, ch>>>(och, ch, devkwt, devU);
    cudaEventRecord(stop);

    gpu_error(cudaFree(devkwt));

    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    overhead_time += milliseconds;

    // float *kwt_new = (float *)malloc(ch*3*3*sizeof(float));
    int n1 = 4, n2 = 4;
    size_t tempusize = ch*4*4*sizeof(float);
    gpu_error(cudaMalloc((void **) & devtempU, tempusize));

    size_t cuttempsize = bs*oph*opw*sizeof(float);
    size_t cutsize = bs*och*oph*opw*sizeof(float);
    cutY = (float *)malloc(cutsize);
    float *cuttempY = (float *)malloc(cuttempsize);

    float *devout, *devsum, *devY, *devcutY;
    float *devfin;
    //devout = devsum = nullptr;
 

    size_t finsize = bs * p * q * ch * 4 * 4 * sizeof(float);
    size_t outsize = bs * p * q * ch * 4 * 4 * sizeof(float);
    size_t sumsize = bs * p * q * 4 * 4 * sizeof(float);
    size_t ysize = bs * p * q * 2 * 2 * sizeof(float);

 
    gpu_error(cudaMalloc((void **) & devout, outsize));
    gpu_error(cudaMalloc((void **) & devsum, sumsize));

    gpu_error(cudaMalloc((void **) & devfin, finsize));
    gpu_error(cudaMalloc((void **) & devY, ysize));
    
    gpu_error(cudaMalloc((void **) & devcutY, cuttempsize));

    LOOP(och)
    {
        cudaEventRecord(start);
        ucopy<<<1,ch>>>(devtempU, devU, toch, n1, n2, ch);
        cudaEventRecord(stop);
        
        cudaEventSynchronize(stop);
        milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        overhead_time += milliseconds;
    
        tilehost(p,q,1,ch,bs,h,w,pad,devin,oph,opw,devtempU,cuttempY,devout, devsum, devfin, devY, devcutY, conv_time, overhead_time);
        LOOP(bs)
            LOOP(oph)
                LOOP(opw)
                    cutY[((((tbs*och+toch)*oph+toph)*opw)+topw)] = cuttempY[(((tbs*oph)+toph)*opw+topw)];
    }
    
    free(cuttempY);

    gpu_error(cudaFree(devcutY));
    gpu_error(cudaFree(devY));
    gpu_error(cudaFree(devout));
    gpu_error(cudaFree(devsum));
    gpu_error(cudaFree(devfin));
    gpu_error(cudaFree(devtempU));    
    gpu_error(cudaFree(devin));    
    gpu_error(cudaFree(devU));

    return cutY;

}