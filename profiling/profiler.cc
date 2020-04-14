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

void analyzeForAlgorithm(DeepNet::Network net, customAlgorithmType t, string dataset_path, std::ofstream &out_log) {
  /* We go upto 8 Batchsize */
  std::cout << "FOR THE ALGORITHM : " << convertAlgoTypeToString(t) << std::endl;
  std::vector<std::vector<profilingElapsedTime>> time_elapsed_conv;
  for(int i = 1; i <= 8; i *= 2) {
    int bs = i;
    std::vector<profilingElapsedTime> time_for_batch;
    int succes = runForwardForBatch(bs, net, t, dataset_path, time_for_batch);
    time_elapsed_conv.push_back(time_for_batch);
  }
  
  for(int i = 0; i < time_elapsed_conv.size(); i++) {
    int bs = pow(2,i);
    out_log << convertAlgoTypeToString(t) << ", ";
    out_log << bs << ", ";
    out_log <<  "TOTAL";
    for(int j = 0; j < time_elapsed_conv[i].size(); j++) {
      out_log << ", " << time_elapsed_conv[i][j].total;
    }
    out_log << "\n";
    out_log << convertAlgoTypeToString(t) << ", ";
    out_log << bs << ", ";
    out_log <<  "CONV";
    for(int j = 0; j < time_elapsed_conv[i].size(); j++) {
      out_log << ", " << time_elapsed_conv[i][j].conv;
    }
    out_log << "\n";
    out_log << convertAlgoTypeToString(t) << ", ";
    out_log << bs << ", ";
    out_log <<  "OVERHEAD";
    for(int j = 0; j < time_elapsed_conv[i].size(); j++) {
      out_log << ", " << time_elapsed_conv[i][j].overhead;
    }
    out_log << "\n";
  }
} 


int main(int argc, char **argv) {
  
  if(argc < 3) {
    std::cerr << "Please provide path to the option - \"VGG\" or \"ALEX\" and respective path to the model" << std::endl;
    exit(1);  
  }

  // vector<customAlgorithmType> algos;
  // string mode(argv[1]);
  // if(mode == "VGG") {
  //   algos = {t_CUSTOM_DIRECT, t_CUSTOM_IM2COL, t_CUSTOM_FFT, t_CUSTOM_WINOGRAD};
  // }
  // else {
  //   algos = {t_CUSTOM_DIRECT, t_CUSTOM_IM2COL,  t_CUSTOM_FFT};
  // }

  customAlgorithmType t = t_CUDNN;
  if(argc == 4) {
    std::string algo(argv[3]);
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
  CNN::loadCNNModelFromFile(argv[2], net);

  string mode(argv[1]);
  std::string log_file_name = "log" + mode + ".txt";
  ofstream my_log_file;
  my_log_file.open (log_file_name.c_str(), std::ios::app);
  if(!my_log_file.is_open()) {
    std::cerr << "Error in Opening File" << std::endl;
    exit(1);
  }

  std::string dataset_path =  "../forward/data/MiniImageNet/";

  // for(customAlgorithmType t : algos) {
  analyzeForAlgorithm(net, t, dataset_path, my_log_file); 
  // }
  my_log_file.close();

  google::protobuf::ShutdownProtobufLibrary();
  
  return 0;
}