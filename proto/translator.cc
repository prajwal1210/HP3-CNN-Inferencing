/************************************************************
 * translator.cc:									                          *
 * Implementation of translator.h                           *
 *                                                          *
 * Author: Tanay Bhartia, Prajwal Singhania                 *
 ************************************************************/

#include "translator.h"

/* Implementation of (Translator)translateConv2D_layer */
Conv2D* Translator::translateConv2D_layer(DeepNet::Layer& layer, cudnnHandle_t cudnn, int input_height, int input_width, int batchsize, customAlgorithmType algo) {
  if(!layer.has_conv()) {
    std::cerr << "Error in translateConv2D_layer: Not a conv2D layer" << std::endl;
    return NULL;
  }
  
  DeepNet::ConvLayer2D conv2D = layer.conv();
  Conv2D* retVal = new Conv2D(conv2D.out_channels(), conv2D.in_channels(), conv2D.height() ,conv2D.width(), 
                batchsize, conv2D.padding(), conv2D.stride(), conv2D.dilation(), input_height, input_width, algo, cudnn);
  
  float* weights = new float[conv2D.weight_size()];
  for(int i = 0; i < conv2D.weight_size(); i++)
    weights[i] = conv2D.weight(i);
  
  retVal->SetWeights(weights);

  if (conv2D.bias_present()) {
    float* biases = new float[conv2D.bias_size()];
    for(int i = 0; i < conv2D.bias_size(); i++)
      biases[i] = conv2D.bias(i);

    retVal->SetBias(biases);
  }

  return retVal;
}

/* Implementation of (Translator)translatePool_layer */
Pool* Translator::translatePool_layer(DeepNet::Layer& layer, cudnnHandle_t cudnn, int batchsize, int in_channels, int input_height, int input_weight) {
  if(!layer.has_pool()) {
    std::cerr << "Error in translatePool_layer: Not a Pool layer" << std::endl;
    return NULL;
  }
  DeepNet::PoolLayer2D pool = layer.pool();
  int type = pool.type() == DeepNet::PoolLayer2D::MAX ? 0 : 1;
  Pool* retVal = new Pool(type, batchsize, in_channels, input_height, input_weight, pool.kernel_size(), pool.padding(), pool.stride(), cudnn);
  return retVal;
}

/* Implementation of (Translator)translateAdaptivePool_layer */
Pool* Translator::translateAdaptivePool_layer(DeepNet::Layer& layer, cudnnHandle_t cudnn, int batchsize, int in_channels, int input_height, int input_weight) {
  if(!layer.has_apool()) {
    std::cerr << "Error in translatePool_layer: Not an Adaptive Pool layer" << std::endl;
    return NULL;
  }
  DeepNet::AdaptivePoolLayer2D apool = layer.apool();
  int type = apool.type() == DeepNet::AdaptivePoolLayer2D::MAX ? 0 : 1;
  Pool* retVal = new Pool(type, batchsize, in_channels, input_height, input_weight, apool.out_y(), apool.out_x(), cudnn);
  return retVal;
}

/* Implementation of (Translator)translateActivation_layer */
Activation* Translator::translateActivation_layer(DeepNet::Layer& layer, cudnnHandle_t cudnn, int batchsize, int in_channels, int input_height, int input_width) {
  if(!layer.has_act()) {
    std::cerr << "Error in translateActivation_layer: Not an Activation layer" << std::endl;
    return NULL;
  }
  const DeepNet::Activation act = layer.act();
  int type = act.type() == DeepNet::Activation::RELU ? 0 : 1;
  Activation* retVal = new Activation(type, batchsize, in_channels, input_height, input_width, cudnn);
  return retVal;
}

/* Implementation of (Translator)translateLinear_layer */
Linear* Translator::translateLinear_layer(DeepNet::Layer& layer, cudnnHandle_t cudnn, cublasHandle_t cublas, int batchsize){
  if(!layer.has_linear()){
    std::cerr << "Error in translateLinear_layer: Not a Linear layer" << std::endl;
    return NULL;
  }
  DeepNet::LinearLayer linear = layer.linear();
  Linear* retVal = new Linear(batchsize, linear.out_nodes(), linear.in_nodes(), cublas);
  
  float* weights = new float[linear.weight_size()];
    for(int i = 0; i < linear.weight_size(); i++)
    weights[i] = linear.weight(i);

  retVal->SetWeights(weights);
  
  if (linear.bias_present()) {
    float* biases = new float[linear.bias_size()];
    for(int i = 0; i < linear.bias_size(); i++)
      biases[i] = linear.bias(i);
    
    retVal->SetBias(biases, cudnn);
  }
  return retVal;
}