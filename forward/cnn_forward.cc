/************************************************************
 * cnn_forward.cc:									                        *
 * Implementation of cnn_forward.h                          *
 *                                                          *
 * Author: Prajwal Singhania								                *
 ************************************************************/

#include "cnn_forward.h"

/* Implementation of loadCNNModelFromFile */
void CNN::loadCNNModelFromFile(const char* model_file, DeepNet::Network& net) {
  std::string model_path(model_file);
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
  net.ParseFromString(messageStr);
}

/* Implementation of forwardPass */
float* CNN::forwardPass(DeepNet::Network net, int& batchsize, int& input_h, int& input_w, int& input_c, float* input, 
                        customAlgorithmType algo, std::vector<profilingElapsedTime>& time_elapsed, bool& succes) {
  Translator T;
  
  cudnnHandle_t cudnn;
  checkCUDNN(cudnnCreate(&cudnn));
  cublasHandle_t cublas;
	checkCudaErrors(cublasCreate(&cublas));

  float* output = input;
  float* prev_output;
  /* Parse the network layer by layer and compute the forward pass */
  for(int i = 0; i < net.layers_size(); i++) {
    DeepNet::Layer net_layer = net.layers(i); 
    std::cout << "Processing the Layer Type : " << DeepNet::Layer_LayerType_Name(net_layer.type());

    switch (net_layer.type()) {
      case DeepNet::Layer::CONV:
        {
          Conv2D* conv = T.translateConv2D_layer(net_layer, cudnn, input_h, input_w, batchsize, algo);
          if(conv == NULL)
          {
            succes = false;
            return NULL;
          }
          std::cout << "(" << conv->out_channels << ", " << conv->in_channels 
                  << ", " << "kernel_size = (" << conv->h  << ", " << conv->w << ")"
                  << ", " << "stride = (" << conv->stride << " ," << conv->stride << ")" 
                  << ", " << "padding = (" << conv->padding << " ," << conv->padding << "))" << " --> ";
          conv->GetOutputDims(&batchsize, &input_c, &input_h, &input_w);
          std::cout << "(" << batchsize << ", " << input_c << ", " << input_h << ", " << input_w << ")" << std::endl;
          profilingElapsedTime time_in_ms;
          prev_output = output;
          output = conv->ConvForward(output, time_in_ms);
          free(prev_output);
          time_elapsed.push_back(time_in_ms);
          break;
        }
      case DeepNet::Layer::POOL:
        {
          Pool* pool = T.translatePool_layer(net_layer, cudnn, batchsize, input_c, input_h, input_w);
          if(pool == NULL)
          {
            succes = false;
            return NULL;
          }
          string t = pool->type == 0 ? "MAX" : "AVG";
          std::cout << "("  << t << ", " << "kernel_size = (" << pool->kernel_size_y  << ", " << pool->kernel_size_x << ")"
                  << ", " << "stride = (" << pool->stride_y << " ," << pool->stride_x << ")" 
                  << ", " << "padding = (" << pool->padding << " ," << pool->padding << "))" << " --> ";
          pool->GetOutputDims(&batchsize, &input_c, &input_h, &input_w);
          std::cout << "(" << batchsize << ", " << input_c << ", " << input_h << ", " << input_w << ")" << std::endl;
          prev_output = output;
          output = pool->PoolForward(output);
          free(prev_output);
          break; 
        }
      case DeepNet::Layer::ADAPTIVE_POOL:
        {
          Pool* pool = T.translateAdaptivePool_layer(net_layer, cudnn, batchsize, input_c, input_h, input_w);
          if(pool == NULL)
          {
            succes = false;
            return NULL;
          }
          string t = pool->type == 0 ? "MAX" : "AVG";
          std::cout << "("  << t << ", " << "kernel_size = (" << pool->kernel_size_y  << ", " << pool->kernel_size_x << ")"
                  << ", " << "stride = (" << pool->stride_y << " ," << pool->stride_x << ")" 
                  << ", " << "padding = (" << pool->padding << " ," << pool->padding << "))" << " --> ";
          pool->GetOutputDims(&batchsize, &input_c, &input_h, &input_w);
          std::cout << "(" << batchsize << ", " << input_c << ", " << input_h << ", " << input_w << ")" << std::endl;
          prev_output = output;
          output = pool->PoolForward(output);
          free(prev_output);
          break; 
        }
      case DeepNet::Layer::ACTIVATION:
        {
          Activation* act = T.translateActivation_layer(net_layer, cudnn, batchsize, input_c, input_h, input_w);
          if(act == NULL)
          {
            succes = false;
            return NULL;
          }
          string t = act->type == 0 ? "RELU" : "SIGMOID";
          std::cout << "("  << t << ")" << " --> ";
          act->GetOutputDims(&batchsize, &input_c, &input_h, &input_w);
          std::cout << "(" << batchsize << ", " << input_c << ", " << input_h << ", " << input_w << ")" << std::endl;
          prev_output = output;
          output = act->ActivationForward(output);
          free(prev_output);
          break;
        }
      case DeepNet::Layer::LINEAR:
        {
          Linear* lin = T.translateLinear_layer(net_layer, cudnn, cublas, batchsize);
          if(lin == NULL)
          {
            succes = false;
            return NULL;
          }
          std::cout << "("  << "out_nodes = " << lin->out_nodes << ", " << "in_nodes = " << lin->in_nodes << ")" << " --> ";
          lin->GetOutputDims(&batchsize, &input_c, &input_h, &input_w);
          std::cout << "(" << batchsize << ", " << input_c << ", " << input_h << ", " << input_w << ")" << std::endl;
          prev_output = output;
          output = lin->LinearForward(output);
          free(prev_output);
          break;
        }
      default:
        std::cout << std::endl;
    }
  }
  succes = true;
  return output;
}
