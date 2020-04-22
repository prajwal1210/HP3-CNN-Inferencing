/************************************************************
 * gpuMemProfiler.cc:									                      *
 * Logger to run the forward pass of the inferencing engine *
 *                                                          *
 * Author: Prajwal Singhania                                *
 ************************************************************/

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>

#include "forward/cnn_forward.h"
#include "forward/data_util.h"

std::string convertAlgoTypeToString(customAlgorithmType t) {
  switch(t) {
    case t_CUDNN:
      return "CUDNN";
    case t_CUSTOM_DIRECT:
      return "DIRECT";
    case t_CUSTOM_FFT:
      return "FFT";
    case t_CUSTOM_WINOGRAD:
      return "WINOGRAD";
    case t_CUSTOM_IM2COL:
      return "IM2COL";
  }
}

int runForwardForBatch(int batchsize, DeepNet::Network net, customAlgorithmType t, string dataset_path, std::vector<profilingElapsedTime>& time_elapsed) {
  int input_h, input_w, input_c;
  bool completed = false;

  MiniImageNetLoader sloader(dataset_path.c_str(), batchsize);
  
  float* input = sloader.loadNextBatch(batchsize, input_c, input_h, input_w, completed);

  std::cout << "Running for Batchsize - " << batchsize << std::endl;

  bool succes = true;

  float* output = CNN::forwardPass(net, batchsize, input_h, input_w, input_c, input, t, time_elapsed, succes);

  if(!succes) {
    std::cerr << "Some error occured in forward pass" << std::endl;
    return 0;
  }
  return 1;
}

void analyzeForAlgorithm(DeepNet::Network net, customAlgorithmType t, string dataset_path) {
  /* We go upto 8 Batchsize */
  std::cout << "FOR THE ALGORITHM : " << convertAlgoTypeToString(t) << std::endl;
  int bs = 1;
  std::vector<profilingElapsedTime> time_for_batch;
  int succes = runForwardForBatch(bs, net, t, dataset_path, time_for_batch);
} 


int main(int argc, char **argv) {
  
  if(argc < 2) {
    std::cerr << "Please provide the path to the model" << std::endl;
    exit(1);  
  }

  customAlgorithmType t = t_CUDNN;
  if(argc >= 3) {
    std::string algo(argv[2]);
    if(algo == "DIRECT") {
      t = t_CUSTOM_DIRECT;
    }
    else if(algo == "FFT") {
      t = t_CUSTOM_FFT;
    }
    else if(algo == "WINOGRAD") {
      t = t_CUSTOM_WINOGRAD;
    }
    else if(algo == "IM2COL") {
      t = t_CUSTOM_IM2COL;
    }

  }
  
  DeepNet::Network net;
  CNN::loadCNNModelFromFile(argv[1], net);

  std::string dataset_path =  "../../forward/data/MiniImageNet/";

  analyzeForAlgorithm(net, t, dataset_path); 

  google::protobuf::ShutdownProtobufLibrary();
  
  return 0;
}