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
  std::vector<profilingElapsedTime> time_elapsed;

  float* output = CNN::forwardPass(net, batchsize, input_h, input_w, input_c, input, t, time_elapsed, succes);

  FILE* fp;
  fp = fopen("final_out.txt" , "w");

  for(int i = 0; i < batchsize * input_c * input_h * input_w; i++)
    fprintf(fp, "%f ",output[i]);
  fprintf(fp, "\n");

  fclose(fp);

  for(auto t : time_elapsed) {
    std::cout << "(" << t.total << ", " << t.conv << "," << t.overhead << "ms), ";
  }
  std::cout << std::endl;

  google::protobuf::ShutdownProtobufLibrary();
  
  return 0;
}