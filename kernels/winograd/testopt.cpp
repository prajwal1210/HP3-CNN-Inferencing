#include <iostream>
#include <random>
#include <algorithm>

#define ACCESS(arr, off, ind) (arr[(off) + (ind)])
#define ACCESS2D(arr, indx, indy) ((arr)[(indx)][(indy)])
#define MAX_B 1
#define MAX_THREAD 1024

using namespace std;

void lastcal1(int och, int p, int q, int bs, float *devsum, float *devY)
{
    int tbs, tp, tq, toch;
    tbs = 0;
    tp = 0;
    tq = 0;
    toch = 0;

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
    printf("%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f \n", ay, by, cy, dy, ey, fy, gy, hy, iy, jy, ky, ly, my, ny, oy, py);
    ind = 0;
    offset = (((tbs*och+toch)*p+tp)*q+tq)*4;
    
    ACCESS(devY, offset, ind++) = ay+ey+iy+by+fy+jy+cy+gy+ky;
    ACCESS(devY, offset, ind++) = by+fy+jy-cy-gy-ky-dy-hy-ly;
    ACCESS(devY, offset, ind++) = ey-iy-my+fy-jy-ny+gy-ky-oy;
    ACCESS(devY, offset, ind++) = fy-jy-ny-gy+ky+oy-hy+ly+py;
}

void lastcal2(int och, int p, int q, int bs, float *devsum, float *devY)
{
    int tbs, tp, tq, toch;
    tbs = 0;
    tp = 0;
    tq = 0;
    toch = 0;

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
    float temp[2][4];
    for(int i = 0; i <2; ++i)
    {
        for(int j = 0; j <4; ++j)
        {
            temp[i][j] = 0;
            for(int k = 0; k <4; ++k)
            {
                temp[i][j] += A_t[i][k] * devsum[((((tbs*och+toch)*p+tp)*q+tq)*4+k)*4+j];
            }
        }
    }
    for(int i = 0; i <2; ++i)
    {
        for(int j = 0; j <2; ++j)
        {
            devY[((((tbs*och+toch)*p+tp)*q+tq)*2+i)*2+j] = 0;
            for(int k = 0; k <4; ++k)
            {
                devY[((((tbs*och+toch)*p+tp)*q+tq)*2+i)*2+j] += temp[i][k] * A[k][j];
            }
        }
    }
}

void tile1(int bs, int p, int q, int ch, float *devin, int h, int w, int och)
{
    int tbs, tp, tq, tch, tbsf, x;
    tbsf = 0;
    tp = 0;
    tq = 0;
    x = 0;
    tbs = tbsf%bs;

    int och_pb = MAX_THREAD/ch;
    int tf = tbsf / bs;
    int toch = x/ch + tf*(och_pb);
    tch = x%ch; 

    float V[4][4];
    
    int offset1 = (tbs*ch + tch)*h*w;
    float av, bv, cv, dv, ev, fv, gv, hv, iv, jv, kv, lv, mv, nv, ov, pv;
    int th = 2*tp, tw = 2*tq;
    av = ACCESS(devin, offset1, th*w + tw++);
    bv = ACCESS(devin, offset1, th*w + tw++);
    cv = ACCESS(devin, offset1, th*w + tw++);
    dv = ACCESS(devin, offset1, th*w + tw++);
    th++; tw = 0;
    ev = ACCESS(devin, offset1, th*w + tw++);
    fv = ACCESS(devin, offset1, th*w + tw++);
    gv = ACCESS(devin, offset1, th*w + tw++);
    hv = ACCESS(devin, offset1, th*w + tw++);
    th++; tw = 0;
    iv = ACCESS(devin, offset1, th*w + tw++);
    jv = ACCESS(devin, offset1, th*w + tw++);
    kv = ACCESS(devin, offset1, th*w + tw++);
    lv = ACCESS(devin, offset1, th*w + tw++);
    th++; tw = 0;
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

    for(int i = 0; i < 4; i++)
    {
        for(int j = 0; j < 4; j++)
            cout<<V[i][j]<<" ";
    }
    // cout<<endl;
}

void tile2(int bs, int p, int q, int ch, float *devin, int h, int w, int och)
{
    int tbs, tp, tq, tch, tbsf, x;
    tbsf = 0;
    tp = 0;
    tq = 0;
    x = 0;
    tbs = tbsf%bs;

    int och_pb = MAX_THREAD/ch;
    int tf = tbsf / bs;
    int toch = x/ch + tf*(och_pb);
    tch = x%ch; 

    float V[4][4];
    float thrtile[4][4];
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

    for(int i = 0; i < 4; i++)
    {
        for(int j = 0; j < 4; j++)
            cout<<V[i][j]<<" ";
    }
    // cout<<endl;
}


int main()
{
    float *devsum, *devy;
    int och, p, q, bs, ch;
    och = p = q = bs = ch = 1;
    devsum = new float[16];
    devy = new float[4];
    auto engine = default_random_engine(time(nullptr));
    auto rng = uniform_real_distribution<float>();
    for(int i = 0; i < 16; i++)
    {
        devsum[i] = rng(engine);
        cout<<devsum[i]<<" ";
    }
    cout<<endl;
    lastcal1(och, p, q, bs, devsum, devy);
    for(int i = 0; i < 4; i++)
    {
        cout<<devy[i]<<" ";
    }
    cout<<endl;
    lastcal2(och, p, q, bs, devsum, devy);
    for(int i = 0; i < 4; i++)
    {
        cout<<devy[i]<<" ";
    }
    cout<<endl;

    tile1(bs, p, q, ch, devsum, 4, 4, 1);
    cout<<endl;

    tile2(bs, p, q, ch, devsum, 4, 4, 1);
    cout<<endl;
    
}