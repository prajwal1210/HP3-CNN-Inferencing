#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

#include "forward/cnn_forward.h"
#include "forward/data_util.h"

int main(int argc, char **argv) {
  
  if(argc != 2) {
    std::cerr << "Please provide path to the model to load relative to the executable directory" << std::endl;
    exit(1);  
  }

  DeepNet::Network net;
  CNN::loadCNNModelFromFile(argv[1], net);
  
  SingleImageLoader sloader;

  int batchsize, input_h, input_w, input_c;
  
  string image_path_s = "../data/n02118333_27_fox.jpg";
  const char* image_path = image_path_s.c_str();
  float* input = sloader.loadSingleColoredImageCHW(image_path, batchsize, input_c, input_h, input_w);
  bool succes = true;

  float* output = CNN::forwardPass(net, batchsize, input_h, input_w, input_c, input, succes);

  FILE* fp;
  fp = fopen("final_out.txt" , "w");

  for(int i = 0; i < batchsize * input_c * input_h * input_w; i++)
    fprintf(fp, "%f ",output[i]);
  fprintf(fp, "\n");

  google::protobuf::ShutdownProtobufLibrary();
  
  return 0;
}