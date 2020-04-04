#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

#include "network.pb.h"

using namespace std;

void printNetwork(const DeepNet::Network& myNetwork){
    if(!myNetwork.has_num_layers()){
        cout<<"This network has no layers."<<endl;
    }
    int num = myNetwork.num_layers();
    int layers = myNetwork.layers_size();
    if(num != layers){
        cout<<"Error in pb file"<<endl;
        return;
    }
    for(int i=0;i<num;i++){
        const DeepNet::Layer& layer = myNetwork.layers(i);
        DeepNet::Layer_LayerType type = layer.type();
        cout<< type <<endl;
    }   
}


int main() {
  string s;
  fstream input("../pretrained-models/vgg19.pb", ios::in | ios::binary);
  int c = 0;

  string str;
  if(input) {
    ostringstream ss;
    ss << input.rdbuf(); // reading data
    str = ss.str();
  }

  DeepNet::Network net;
  net.ParseFromString(str);


  printNetwork(net);

  google::protobuf::ShutdownProtobufLibrary();
}