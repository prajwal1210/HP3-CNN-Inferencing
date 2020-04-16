#include "winograd_mem.cu"

#include<random>
#define LOOP(x) for(int t##x = 0; t##x < x; t##x++)
using namespace std;

int main(void) 
{
    auto engine = default_random_engine(7);
    auto rng = uniform_real_distribution<float>();
    int bs, ch, h, w, och, pad; 
    //bs - batch size, ch - input channel, h - input height, w - input weight , pad - padding required
    // bs = 1;
    // ch = 64;
    // h = 32;
    // w = 32;
    // och = 10;
    // pad = 0;
    // bs = 8;
    // ch = 16;
    // h = 256;
    // w = 256;
    // och = 64;
    // pad = 0;
     bs = 2;
    ch = 2;
    h = 5;
    w = 5;
    och = 2;
    pad = 0;
    size_t insize = bs * ch * h * w * sizeof(float);
    float *in = new float[insize/sizeof(float)];
    float *t = in;
    float  *cutY; //final convolved output
    size_t tsize = och*ch*3*3;
    float *kernel_weights = new float[tsize];
    float *tkw = kernel_weights;
    //put kernel weights
    
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
     LOOP(tsize)
    {
        tkw[ttsize] = rng(engine);
    }
   tkw[0] = tkw[8] = 1;
    tkw[9] = tkw[17] = 2;
    tkw[18] = tkw[26] = 3;
    tkw[27] = tkw[35] = 4;

   // cout<<"\nConvolving\n";
    
    int oph, opw; //output height, output weight
    cutY = WING::forward(och, ch, bs, h, w, pad, in, oph, opw, kernel_weights);

    //cout<<"\nConvolution finished\n\n";
      
    LOOP(bs)
    {
        cout<<"{ ";
        LOOP(och)
        {
            cout<<"{ ";
            LOOP(oph)
            {
                LOOP(opw)
                {
                    cout<<cutY[((tbs*och+toch)*oph+toph)*opw+topw]<<",";
                }
                cout<<";\n";
            }
            cout<<"}\n";
        }
        cout<<"}\n";
    }
    cout<<"}\n";
    // int n1 = 4, n2 = 4;
    //  LOOP(och)
    // {
    //     cout<<"{ ";
    //     LOOP(ch)
    //     {
    //         cout<<"{ ";
    //         LOOP(n1)
    //         {
    //             LOOP(n2)
    //             {
    //                 cout<<cutY[((toch*ch+tch)*n1+tn1)*n2+tn2]<<",";
    //             }
    //             cout<<";\n";
    //         }
    //         cout<<"}\n";
    //     }
    //     cout<<"}\n";
    // }
    // cout<<"}\n";

    // int p = 2, q = 2;
    // // ch = 2;

    // // (((((tbs*och+toch)*p+tp)*q+tq)*ch+tch)*n1 + tn1)*n2 + tn2
    // // (((((tbs*p+tp)*q+tq)*ch+tch)*och+toch)*n1+tn1)*n2+tn2

    // LOOP(bs)
    // {
    //     cout<<"{ ";
    //     LOOP(och)
    //     {
    //         cout<<"{ ";
    //         LOOP(p)
    //         {
    //             cout<<"{ ";
    //             LOOP(q)
    //             {
    //                 cout<<"{ ";
    //                 LOOP(ch)
    //                 {
    //                     cout<<"{ ";
    //                     LOOP(n1)
    //                         {
    //                             cout<<"{ ";
    //                             LOOP(n2)
    //                             {   
    //                                 cout<<cutY[(((((tbs*och+toch)*p+tp)*q+tq)*1+0)*n1 + tn1)*n2 + tn2]<<" ";
    //                             }
    //                         cout<<"}\n";
    //                         }
    //                     cout<<"}\n";
    //                 }
    //                 cout<<"}\n";
    //                 break;
    //             }
    //             cout<<"}\n";
    //                 break;

    //         }
    //     cout<<"}\n";
    //                 break;

    //     }
    // cout<<"}\n";
    //                 break;

    // }
    return 0;
}