#include <iostream>
#include <fstream>
#include <string>
#include "network.pb.h"

using namespace std;

void printNetwork(const DeepNet::Network& myNetwork){
    if(!myNetwork.has_num_layers()){
        cout<<"This network has no layers."<<endl;
    }
    int num = myNetwork.num_layers();
}

int main(int argc, char* argv[]){
    GOOGLE_PROTOBUF_VERIFY_VERSION;
    DeepNet::Network myNetwork;

    fstream input(argv[0], ios::in | ios::binary);
    if (!myNetwork.ParseFromIstream(&input)) {
      cerr << "Failed to parse address book." << endl;
      return -1;
    }    
    return 0;
}