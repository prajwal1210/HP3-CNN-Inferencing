/************************************************************
 * cnn_forward.h:									        *
 * Library that provides functions to load a pretrained CNN *
 * from a file and a function to run forward pass given the *
 * network and input runs a forward                         *
 * pass on it                                               *
 *                                                          *
 * Author: Prajwal Singhania								*
 ************************************************************/

#ifndef CNN_FORWARD_H 
#define CNN_FORWARD_H

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

#include "forward/operations.h"
#include "proto/network.pb.h"
#include "proto/translator.h"

namespace CNN {
    void loadCNNModelFromFile(const char* model_file, DeepNet::Network& net);
    float* forwardPass(DeepNet::Network net, int& batchsize, int& input_h, int& input_w, int& input_c, float* input, bool& succes);
}
#endif