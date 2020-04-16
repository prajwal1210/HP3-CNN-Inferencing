#include "wingheader.h"

#define MAX_B 1
#define MAX_THREAD 1024
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
#define ACCESS(arr, off, ind) ((arr)[(off) + (ind)])
#define ACCESS2D(arr, indx, indy) ((arr)[(indx)][(indy)])


void gpu_error(cudaError_t const &code) {
    if(code != cudaSuccess)
    {
        std::cerr << "GPUError: Code " << code << " : " << cudaGetErrorString(code) << std::endl;
        exit( EXIT_FAILURE );
    }
}

__global__ void precompute(int och, int ch, float* kernel_weights, float *U)
{
    int tch = blockIdx.x;
    int toch = threadIdx.x;
   
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
    float temp[3][4];
    for(int i = 0; i <3; ++i)
    {
        for(int j = 0; j <4; ++j)
        {
            temp[i][j] = 0;
            for(int k = 0; k <3; ++k)
            {
                temp[i][j] += kernel_weights[((toch*ch + tch)*3 + i)*3+k] * g_t[k][j];
            }
        }
    }
    for(int i = 0; i <4; ++i)
    {
        for(int j = 0; j <4; ++j)
        {
            U[((toch*ch + tch)*4 + i)*4+j] = 0;
            for(int k = 0; k <3; ++k)
            {
                U[((toch*ch + tch)*4 + i)*4+j] += g[i][k] * temp[k][j];
            }
        }
    }

}


__global__ void paddev(float *devin, float *devinnopad, int h, int w, int pad)
{
    int newh = gridDim.y;
    int neww = gridDim.z;
    int tbs = blockIdx.x;
    int tch = threadIdx.x;
    int ch = blockDim.x;
    int tnewh = blockIdx.y;
    int tneww = blockIdx.z;
    int newhw = newh*neww;
    int hw = h*w;
    int th = tnewh-pad;
    int tw = tneww-pad;
    int tbsch = tbs*ch + tch;
    
    if(th >= 0 && th < h && tw >= 0 && tw < w)
        devin[tbsch*newhw + tnewh*neww + tneww] = devinnopad[tbsch*hw + th*w + tw];
    else
        devin[tbsch*newhw + tnewh*neww + tneww] = 0;
    
}

__global__ void cutpad(float  *devY, float *devcutY, int oph,int opw)
{
    int p = gridDim.y;
    int q = gridDim.z;
    int tbs = blockIdx.x;
    int tp = blockIdx.y;
    int tq = blockIdx.z;
    int toch = threadIdx.x;
    int och = blockDim.x;
    int offset = tbs*och+toch;
    int ophopw = oph*opw;
    for(int i = 0; i < 2; i++)
    {
        for(int j = 0; j < 2; j++)
        {
            if(tp*2 + i < oph && tq*2 + j < opw)
                devcutY[offset*ophopw + (tp*2+i)*opw + (tq*2+j)] = devY[(((offset*p + tp)*q +tq)*2 + i)*2  + j];
        }
    }
}

__global__ void tile(int bs, int p, int q, int ch, float *devin, float *devsum, float *devU, int h, int w, int och, float *devfin)
{
    float thrtile[4][4];    
    int tbs, tp, tq, tch, Tch;
    tbs = blockIdx.x;
    tp = blockIdx.y;
    tq = blockIdx.z;
    Tch = threadIdx.x;
    float V[4][4];
    tch = Tch / och;
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
    __syncthreads();

    int toch = Tch % och;
    tch = Tch / och;

    for(int i = 0; i <4; ++i)
        for(int j = 0; j <4; ++j)
            devfin[(((((tbs*p+tp)*q+tq)*ch+tch)*och+toch)*4+i)*4+j] = devU[((toch*ch+tch)*4+i)*4+j]*V[i][j]; 
    
    __syncthreads();

    for(int s = 1; s < ch; s *= 2)
    {
        if(tch % (2*s) == 0 && tch+s < ch)
        {
            toch = Tch % och;
            // LOOP(och)
                for(int i = 0; i < 4; i++)
                    for(int j = 0; j < 4; j++)
                        devfin[(((((tbs*p+tp)*q+tq)*ch+tch)*och+toch)*4+i)*4+j] += devfin[(((((tbs*p+tp)*q+tq)*ch+(tch+s))*och+toch)*4+i)*4+j];
        }
        __syncthreads();
    }

    if(tch == 0) 
    {

            for(int i = 0; i < 4; i++)
                for(int j = 0; j < 4; j++)
                    devsum[((((tbs*och+toch)*p+tp)*q+tq)*4 + i)*4 + j] = devfin[(((((tbs*p+tp)*q+tq)*ch+0)*och+toch)*4+i)*4+j];
    }
     __syncthreads();
  
}

__global__ void tile2(int bs, int p, int q, int ch, float *devin, float *devsum, float *devU, int h, int w, int och, float *devfin)
{
    int tbs, tp, tq, tch, tbsf, x;
    tbsf = blockIdx.x;
    tp = blockIdx.y;
    tq = blockIdx.z;
    x = threadIdx.x;
    tbs = tbsf%bs;

    int och_pb = MAX_THREAD/ch;
    int tf = tbsf / bs;
    int toch = x/ch + tf*(och_pb);
    tch = x%ch; 

    float V[4][4];
    
    // copy the tiles to thrtile
    // int offset1 = (tbs*ch + tch)*h*w;
    // for(int th = 2*tp, i = 0; i < 4; th++, i++)
    //     for(int tw = 2*tq, j = 0; j < 4; tw++, j++)
    //         thrtile[i][j] = devin[offset1 + th*w + tw];

    int offset1 = (tbs*ch + tch)*h*w;
    float av, bv, cv, dv, ev, fv, gv, hv, iv, jv, kv, lv, mv, nv, ov, pv;
    int th = 2*tp, tw = 2*tq;
    av = ACCESS(devin, offset1, th*w + tw++);
    bv = ACCESS(devin, offset1, th*w + tw++);
    cv = ACCESS(devin, offset1, th*w + tw++);
    dv = ACCESS(devin, offset1, th*w + tw++);
    th++; tw = 2*tq;
    ev = ACCESS(devin, offset1, th*w + tw++);
    fv = ACCESS(devin, offset1, th*w + tw++);
    gv = ACCESS(devin, offset1, th*w + tw++);
    hv = ACCESS(devin, offset1, th*w + tw++);
    th++; tw = 2*tq;
    iv = ACCESS(devin, offset1, th*w + tw++);
    jv = ACCESS(devin, offset1, th*w + tw++);
    kv = ACCESS(devin, offset1, th*w + tw++);
    lv = ACCESS(devin, offset1, th*w + tw++);
    th++; tw = 2*tq;
    mv = ACCESS(devin, offset1, th*w + tw++);
    nv = ACCESS(devin, offset1, th*w + tw++);
    ov = ACCESS(devin, offset1, th*w + tw++);
    pv = ACCESS(devin, offset1, th*w + tw++);
    
    //Calculation of V
    int vx = 0, vy = 0;
    ACCESS2D(V, vx, vy++) = +av-iv-cv+kv;
    ACCESS2D(V, vx, vy++) = +bv-jv+cv-kv;
    ACCESS2D(V, vx, vy++) = -bv+jv+cv-kv;
    ACCESS2D(V, vx, vy++) = +bv-jv-dv+lv;
    vx++; vy = 0;
    ACCESS2D(V, vx, vy++) = +ev+iv-gv-kv;
    ACCESS2D(V, vx, vy++) = +fv+jv+gv+kv;
    ACCESS2D(V, vx, vy++) = -fv-jv+gv+kv;
    ACCESS2D(V, vx, vy++) = +fv+jv-hv-lv;
    vx++; vy = 0;
    ACCESS2D(V, vx, vy++) = -ev+iv+gv-kv;
    ACCESS2D(V, vx, vy++) = -fv+jv-gv+kv;
    ACCESS2D(V, vx, vy++) = +fv-jv-gv+kv;
    ACCESS2D(V, vx, vy++) = -fv+jv+hv-lv;
    vx++; vy = 0;
    ACCESS2D(V, vx, vy++) = +ev-mv-gv+ov;
    ACCESS2D(V, vx, vy++) = +fv-nv+gv-ov;
    ACCESS2D(V, vx, vy++) = -fv+nv+gv-ov;
    ACCESS2D(V, vx, vy++) = +fv-nv-hv+pv;
    
    __syncthreads();

    for(int i = 0; i <4; ++i)
        for(int j = 0; j <4; ++j)
            devfin[(((((tbs*p+tp)*q+tq)*ch+tch)*och+toch)*4+i)*4+j] = devU[((toch*ch+tch)*4+i)*4+j]*V[i][j]; 
    
    __syncthreads();

    for(int s = 1; s < ch; s *= 2)
    {
        if(tch % (2*s) == 0 && tch+s < ch)
        {
            //toch = Tch % och;
            // LOOP(och)
                for(int i = 0; i < 4; i++)
                    for(int j = 0; j < 4; j++)
                        devfin[(((((tbs*p+tp)*q+tq)*ch+tch)*och+toch)*4+i)*4+j] += devfin[(((((tbs*p+tp)*q+tq)*ch+(tch+s))*och+toch)*4+i)*4+j];
        }
        __syncthreads();
    }

    if(tch == 0) 
    {

            for(int i = 0; i < 4; i++)
                for(int j = 0; j < 4; j++)
                    devsum[((((tbs*och+toch)*p+tp)*q+tq)*4 + i)*4 + j] = devfin[(((((tbs*p+tp)*q+tq)*ch+0)*och+toch)*4+i)*4+j];
    }
     __syncthreads();
  
}

__global__ void lastcal(int och, int p, int q, int bs, float *devsum, float *devY)
{
    int tbs, tp, tq, toch;
    tbs = blockIdx.x;
    tp = blockIdx.y;
    tq = blockIdx.z;
    toch = threadIdx.x;

    int offset = (((tbs*och+toch)*p+tp)*q+tq)*16;
    float ay, by, cy, dy, ey, fy, gy, hy, iy, jy, ky, ly, my, ny, oy, py;
    int ind = 0;
    ay = ACCESS(devsum, offset, ind++);
    by = ACCESS(devsum, offset, ind++);
    cy = ACCESS(devsum, offset, ind++);
    dy = ACCESS(devsum, offset, ind++);
    ey = ACCESS(devsum, offset, ind++);
    fy = ACCESS(devsum, offset, ind++);
    gy = ACCESS(devsum, offset, ind++);
    hy = ACCESS(devsum, offset, ind++);
    iy = ACCESS(devsum, offset, ind++);
    jy = ACCESS(devsum, offset, ind++);
    ky = ACCESS(devsum, offset, ind++);
    ly = ACCESS(devsum, offset, ind++);
    my = ACCESS(devsum, offset, ind++);
    ny = ACCESS(devsum, offset, ind++);
    oy = ACCESS(devsum, offset, ind++);
    py = ACCESS(devsum, offset, ind++);
    
    ind = 0;
    offset = (((tbs*och+toch)*p+tp)*q+tq)*4;
    
    ACCESS(devY, offset, ind++) = ay+ey+iy+by+fy+jy+cy+gy+ky;
    ACCESS(devY, offset, ind++) = by+fy+jy-cy-gy-ky-dy-hy-ly;
    ACCESS(devY, offset, ind++) = ey-iy-my+fy-jy-ny+gy-ky-oy;
    ACCESS(devY, offset, ind++) = fy-jy-ny-gy+ky+oy-hy+ly+py;
}


float * WING::forward(int och, int ch, int bs, int h, int w, int pad, float *in, int &oph, int &opw, float *kwt, float& conv_time, float& overhead_time)
{
    conv_time = 0;
    overhead_time = 0;
    float milliseconds = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float *devin, *devinnopad, *cutY, *devkwt, *devU;
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
    dim3 padgrid(bs, newh, neww);
    dim3 padblock(ch, 1, 1);
    
    cudaEventRecord(start);
    paddev<<<padgrid,padblock>>>(devin, devinnopad, h, w, pad);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    overhead_time += milliseconds;

    gpu_error(cudaFree(devinnopad));
    h = newh;
    w = neww;

    size_t kwtsize = och*ch*3*3*sizeof(float);    
    size_t usize = och*ch*4*4*sizeof(float);
    gpu_error(cudaMalloc((void **) & devkwt, kwtsize));
    gpu_error(cudaMalloc((void **) & devU, usize));
    gpu_error(cudaMemcpy(devkwt, kwt, kwtsize, cudaMemcpyHostToDevice));

    cudaEventRecord(start);
    precompute<<<ch, och>>>(och, ch, devkwt, devU);
    cudaEventRecord(stop);

    gpu_error(cudaFree(devkwt));

    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    overhead_time += milliseconds;

    size_t cutsize = bs*och*oph*opw*sizeof(float);
    cutY = (float *)malloc(cutsize);

    float *devsum, *devY, *devcutY;
    float *devfin;
    int p = max((h-2)/2, 0);
    int q = max((w-2)/2, 0);

    size_t finsize = MAX_B * p * q * ch * och * 4 * 4 * sizeof(float);
    size_t sumsize = bs * och * p * q * 4 * 4 * sizeof(float);
    size_t ysize = bs * och * p * q * 2 * 2 * sizeof(float);

 
    gpu_error(cudaMalloc((void **) & devsum, sumsize));

    gpu_error(cudaMalloc((void **) & devfin, finsize));

    size_t binsize = ch * newh * neww ;
    size_t dsumsize = och * p * q * 4 * 4 ;
    int bsg = (bs+MAX_B-1)/MAX_B;
    int prevb = 0;
    LOOP(bsg)
    {
        int currb = MAX_B;
        if(tbsg == bsg-1 && bs % MAX_B != 0)
            currb = bs % MAX_B;
        if(och*ch <= MAX_THREAD)
        {
            dim3 grid(currb, p, q); 
            dim3 block(och*ch, 1, 1);
            cudaEventRecord(start);
            tile<<<grid, block>>>(currb, p, q, ch, devin + prevb*binsize, devsum + prevb*dsumsize, devU, h, w, och, devfin);
            cudaEventRecord(stop);

            cudaEventSynchronize(stop);
            milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);
            conv_time += milliseconds;

        }
        else
        {
            int f = (och*ch)/MAX_THREAD;
            dim3 grid(currb*f, p, q); 
            dim3 block(MAX_THREAD, 1, 1);
            cudaEventRecord(start);
            tile2<<<grid, block>>>(currb, p, q, ch, devin + prevb*binsize, devsum + prevb*dsumsize, devU, h, w, och, devfin);   
            cudaEventRecord(stop);

            cudaEventSynchronize(stop);
            milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);
            conv_time += milliseconds;

        }
        prevb  += currb;
    }

    gpu_error(cudaFree(devfin));
    gpu_error(cudaFree(devin));    
    gpu_error(cudaFree(devU));

    dim3 grid2(bs, p, q);
    dim3 block2(och, 1, 1);
    gpu_error(cudaMalloc((void **) & devY, ysize));
    cudaEventRecord(start);
    lastcal<<<grid2,block2>>>(och, p, q, bs, devsum, devY);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    overhead_time += milliseconds;

    gpu_error(cudaFree(devsum));

    dim3 cutgrid(bs, p, q);
    dim3 cutblock(och,1,1);
    
    
    gpu_error(cudaMalloc((void **) & devcutY, cutsize));
    cudaEventRecord(start);
    cutpad<<<cutgrid, cutblock>>> (devY, devcutY, oph, opw);  
    cudaEventRecord(stop);

 
    gpu_error(cudaFree(devY));

    cudaSafeCall(cudaMemcpy(cutY, devcutY, cutsize, cudaMemcpyDeviceToHost));
    
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    overhead_time += milliseconds;

    gpu_error(cudaFree(devcutY));

    gpu_error(cudaEventDestroy(start));
    gpu_error(cudaEventDestroy(stop));

    return cutY;

}