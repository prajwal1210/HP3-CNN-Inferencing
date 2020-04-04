#include "translator.h"
#include <stdio.h>
#include <iostream>

using namespace std;

Conv2D Translator::translateConv2D_layer(DeepNet::Layer& layer, cudnnHandle_t cudnn,int h,int w, int batchsize){
    if(!layer.has_conv()){
        cout<<"Error in translateConv2D_layer: Not a conv2D layer"<<endl;
        return null;
    }
    const DeepNet::ConvLayer2D conv2D = layer.conv();
    Conv2D retVal = new Conv2D(conv2D.out_channels(), conv2D.in_channels(),h,w,batchsize,conv2D.padding(),conv2D.stride(),conv2D.dilation(),conv2D.height(),conv2D.weight(),cudnn);
    return retVal;
}

Pool Translator::translatePool_layer(DeepNet::Layer& layer, cudnnHandle_t cudnn,int batchsize,int in_channels,int input_height,int input_weight){
    if(!layer.has_pool()){
        cout<<"Error in translatePool_layer: Not a Pool layer"<<endl;
        return null;
    }
    const DeepNet::PoolLayer2D pool = layer.pool();
    int type = pool.type() == DeepNet::PoolLayer2D_PoolType.PoolLayer2D_PoolType_MAX ? 0 : 1;
    Pool retVal = new Pool(type,batchsize,in_channels,input_height,input_weight,pool.kernel_size(),pool.padding(),pool.stride(),cudnn);
    return retVal;
}

Activation Translator::translateActivation_layer(DeepNet::Layer& layer, cudnnHandle_t cudnn, int batchsize,int in_channels,int input_height,int input_weight){
    if(!layer.has_act()){
        cout<<"Error in translateActivation_layer: Not an Activation layer"<<endl;
        return null;
    }
    const DeepNet::Activation act = layer.act();
    int type = act.type() == DeepNet::Activation_ActivationType.Activation_ActivationType_RELU ? 0:1;
    Activation retVal = new Activation(type,batchsize,in_channels,input_height,input_width,cudnn);
    return retVal;
}

Linear Translator::translateLinear_layer(DeepNet::Layer& layer, cudnnHandle_t cudnn, int batchsize){
    if(!layer.has_linear()){
        cout<<"Error in translateLinear_layer: Not a Linear layer"<<endl;
        return null;
    }
    const DeepNet::LinearLayer linear = layer.linear();
    Linear retVal = new Linear(batchsize,linear.out_nodes(),linear.in_nodes(),cudnn);
    return retVal;
}