/************************************************************
 * forward.cc:									                            *
 * Loads a pretrained network and runs a forward pass on it *
 *                                                          *
 * Author: Prajwal Singhania								                *
 ************************************************************/

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

#include "proto/network.pb.h"
#include "proto/translator.h"
#include "forward/operations.h"

int main(int argc, char **argv) {
  
  if(argc != 2) {
    std::cerr << "Please provide path to the model to load relative to the executable directory" << std::endl;
    exit(1);  
  }

  std::string model_path(argv[1]);
  std::fstream inFile(model_path, std::ios::in | std::ios::binary);
  std::string messageStr;

  if(inFile) {
    std::ostringstream ostream;
    ostream  << inFile.rdbuf();
    messageStr = ostream.str();
  }
  else {
    std::cerr << model_path << " not found" << std::endl;
    exit(1);
  }

  DeepNet::Network net;
  net.ParseFromString(messageStr);
  
  cudnnHandle_t cudnn;
  checkCUDNN(cudnnCreate(&cudnn));
  cublasHandle_t cublas;
	checkCudaErrors(cublasCreate(&cublas));

  Translator T;

  int batchsize = 1;
  int input_h = 256;
  int input_w = 256;
  int input_c = 3;

  /* Parse the network layer by layer and compute the forward pass */
  for(int i = 0; i < net.layers_size(); i++) {
    DeepNet::Layer net_layer = net.layers(i); 
    std::cout << "Processing the Layer Type : " << DeepNet::Layer_LayerType_Name(net_layer.type());

    switch (net_layer.type()) {
      case DeepNet::Layer::CONV:
        {
          Conv2D* conv = T.translateConv2D_layer(net_layer, cudnn, input_h, input_w, batchsize);
          std::cout << "(" << conv->out_channels << ", " << conv->in_channels 
                  << ", " << "kernel_size = (" << conv->h  << ", " << conv->w << ")"
                  << ", " << "stride = (" << conv->stride << " ," << conv->stride << ")" 
                  << ", " << "padding = (" << conv->padding << " ," << conv->padding << "))" << " --> ";
          conv->GetOutputDims(&batchsize, &input_c, &input_h, &input_w);
          std::cout << "(" << batchsize << ", " << input_c << ", " << input_h << ", " << input_w << ")" << std::endl;
          break;
        }
      case DeepNet::Layer::POOL:
        {
          Pool* pool = T.translatePool_layer(net_layer, cudnn, batchsize, input_c, input_h, input_w);
          std::cout << "("  << "kernel_size = (" << pool->kernel_size_y  << ", " << pool->kernel_size_x << ")"
                  << ", " << "stride = (" << pool->stride_y << " ," << pool->stride_x << ")" 
                  << ", " << "padding = (" << pool->padding << " ," << pool->padding << "))" << " --> ";
          pool->GetOutputDims(&batchsize, &input_c, &input_h, &input_w);
          std::cout << "(" << batchsize << ", " << input_c << ", " << input_h << ", " << input_w << ")" << std::endl;
          break; 
        }
      case DeepNet::Layer::ACTIVATION:
        {
          Activation* act = T.translateActivation_layer(net_layer, cudnn, batchsize, input_c, input_h, input_w);
          string t = act->type == 0 ? "RELU" : "SIGMOID";
          std::cout << "("  << t << ")" << " --> ";
          act->GetOutputDims(&batchsize, &input_c, &input_h, &input_w);
          std::cout << "(" << batchsize << ", " << input_c << ", " << input_h << ", " << input_w << ")" << std::endl;
          break;
        }
      case DeepNet::Layer::LINEAR:
        {
          Linear* lin = T.translateLinear_layer(net_layer, cudnn, cublas, batchsize);
          std::cout << "("  << "out_nodes = " << lin->out_nodes << ", " << "in_nodes = " << lin->in_nodes << ")" << " --> ";
          lin->GetOutputDims(&batchsize, &input_c, &input_h, &input_w);
          std::cout << "(" << batchsize << ", " << input_c << ", " << input_h << ", " << input_w << ")" << std::endl;
          break;
        }
      default:
        std::cout << std::endl;
    }
  }
  google::protobuf::ShutdownProtobufLibrary();
  
  return 0;
}