#include <iostream>
#include <algorithm>
#include <random>

#define LOOP(x) for(int t##x = 0; t##x < x; t##x++)

using namespace std;

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

int main()
{
    auto engine = default_random_engine(time(nullptr));
    auto rng = uniform_real_distribution<float>();
    int ch = 4, bs = 5, h = 2, w = 3;
    float *in = new float[bs * ch * h * w];
    float *iter = in;
    LOOP(bs)
    {
        LOOP(ch)
        {
            LOOP(h)
            {
                LOOP(w)
                {
                    *(iter++) = rng(engine);
                }
            }
        }
    }
    
    LOOP(bs)
    {
        LOOP(ch)
        {
            LOOP(h)
            {
                LOOP(w)
                {
                    cout<<in[tbs*ch*h*w + tch*h*w + th*w + tw]<<" ";
                }
                cout<<"\t";
            }
            cout<<endl;
        }
        cout<<endl;
    }
    cout<<endl;

    rearrange(ch, bs, h, w, in);
    
    LOOP(ch)
    {
        LOOP(h)
        {
            LOOP(w)
            {
                LOOP(bs)
                {
                    cout<<in[tch*h*w*bs + th*w*bs + tw*bs + tbs]<<" ";
                }
                cout<<"\t";
            }
            cout<<endl;
        }
        cout<<endl;
    }
    cout<<endl;
    
    return 0;
}