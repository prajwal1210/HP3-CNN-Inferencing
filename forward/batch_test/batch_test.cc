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
  

  int batchsize, input_h, input_w, input_c;
  bool completed = false;
  string dataset_path = "../data/MiniImageNet/";
  MiniImageNetLoader sloader(dataset_path.c_str(), 8);
  float* input = sloader.loadNextBatch(batchsize, input_c, input_h, input_w, completed);
  std::cout << "Input Size - (" << batchsize << ", " << input_c << ", " << input_h << ", " << input_w << ")" << std::endl;

  if(batchsize == 0) {
    std::cout << "No Batch" << endl;
    return 0;
  }
  
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