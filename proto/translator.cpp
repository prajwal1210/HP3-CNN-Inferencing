#include "translator.h"
#include <stdio.h>
#include <iostream>
#include <vector>
#include <typeinfo>


using namespace std;

Conv2D* Translator::translateConv2D_layer(DeepNet::Layer& layer, cudnnHandle_t cudnn, int input_height, int input_width, int batchsize) {
  if(!layer.has_conv()) {
    cout << "Error in translateConv2D_layer: Not a conv2D layer" << endl;
    return NULL;
  }
  
  DeepNet::ConvLayer2D conv2D = layer.conv();
  Conv2D* retVal = new Conv2D(conv2D.out_channels(), conv2D.in_channels(),conv2D.height() ,conv2D.width(), 
                batchsize, conv2D.padding(), conv2D.stride(), conv2D.dilation(), input_height, input_width, cudnn);
  float* weights = conv2D.mutable_weight()->mutable_data();
  retVal->SetWeights(weights);
  
  if (conv2D.bias_present()) {
    float* biases = conv2D.mutable_bias()->mutable_data();
    retVal->SetBias(biases);
  }

  return retVal;
}

Pool* Translator::translatePool_layer(DeepNet::Layer& layer, cudnnHandle_t cudnn, int batchsize, int in_channels, int input_height, int input_weight) {
  if(!layer.has_pool()) {
    cout << "Error in translatePool_layer: Not a Pool layer" << endl;
    return NULL;
  }
  DeepNet::PoolLayer2D pool = layer.pool();
  int type = pool.type() == DeepNet::PoolLayer2D::MAX ? 0 : 1;
  Pool* retVal = new Pool(type, batchsize, in_channels, input_height, input_weight, pool.kernel_size(), pool.padding(), pool.stride(), cudnn);
  return retVal;
}

Activation* Translator::translateActivation_layer(DeepNet::Layer& layer, cudnnHandle_t cudnn, int batchsize, int in_channels, int input_height, int input_width) {
  if(!layer.has_act()) {
    cout<<"Error in translateActivation_layer: Not an Activation layer"<<endl;
    return NULL;
  }
  const DeepNet::Activation act = layer.act();
  int type = act.type() == DeepNet::Activation::RELU ? 0 : 1;
  Activation* retVal = new Activation(type, batchsize, in_channels, input_height, input_width, cudnn);
  return retVal;
}

Linear* Translator::translateLinear_layer(DeepNet::Layer& layer, cudnnHandle_t cudnn, cublasHandle_t cublas, int batchsize){
  if(!layer.has_linear()){
    cout << "Error in translateLinear_layer: Not a Linear layer" << endl;
    return NULL;
  }
  DeepNet::LinearLayer linear = layer.linear();
  Linear* retVal = new Linear(batchsize, linear.out_nodes(), linear.in_nodes(), cublas);
  float* weights = linear.mutable_weight()->mutable_data();
  retVal->SetWeights(weights);
  
  if (linear.bias_present()) {
    float* biases = linear.mutable_bias()->mutable_data();
    retVal->SetBias(biases, cudnn);
  }
  return retVal;
}