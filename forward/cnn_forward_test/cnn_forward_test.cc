#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>

#include "forward/cnn_forward.h"
#include "forward/data_util.h"

int main(int argc, char **argv) {
  
  if(argc < 2) {
    std::cerr << "Please provide path to the model to load relative to the executable directory" << std::endl;
    exit(1);  
  }

  customAlgorithmType t = t_CUDNN;
  if(argc == 3) {
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
  
  SingleImageLoader sloader;

  int batchsize, input_h, input_w, input_c;
  
  string image_path_s = "../data/sample_fox.png";
  const char* image_path = image_path_s.c_str();
  float* input = sloader.loadSingleColoredImageCHW(image_path, batchsize, input_c, input_h, input_w);
  bool succes = true;
  std::vector<float> time_conv;

  float* output = CNN::forwardPass(net, batchsize, input_h, input_w, input_c, input, t, time_conv, succes);
  
  FILE* fp;
  fp = fopen("final_out.txt" , "w");

  for(int i = 0; i < batchsize * input_c * input_h * input_w; i++)
    fprintf(fp, "%f ",output[i]);
  fprintf(fp, "\n");

  fclose(fp);

  for(auto t : time_conv) {
    std::cout << t << "ms, ";
  }
  std::cout << std::endl;

  google::protobuf::ShutdownProtobufLibrary();
  
  return 0;
}