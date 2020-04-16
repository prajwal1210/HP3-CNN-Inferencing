#include <iostream>
#include <random>
#include <algorithm>

#define ACCESS(arr, off, ind) (arr[(off) + (ind)])

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

int main()
{
    float *devsum, *devy;
    int och, p, q, bs;
    och = p = q = bs = 1;
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
}