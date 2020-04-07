/************************************************************
 * cnn_forward.h:									                          *
 * Library that provides functions to load a pretrained CNN *
 * from a file and a function to run forward pass given the *
 * network and input runs a forward                         *
 * pass on it                                               *
 *                                                          *
 * Author: Prajwal Singhania								                *
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
  /* Loads the CNN Model from the specification file into the provided object reference
   * Input  : Path to the model, Reference to DeepNet::Network object
   */
  void loadCNNModelFromFile(const char* model_file, DeepNet::Network& net);
  
  /* Iterates over the CNN Network and computes the forward pass on the provided data
   * Input  : DeepNet::Network object, references to dimensions of the input, pointer to the input array, reference to the status flag
   * Output : Pointer to the final layer output (side effect - sets the dimensions accordingly)
   */
  float* forwardPass(DeepNet::Network net, int& batchsize, int& input_h, int& input_w, int& input_c, float* input, bool& succes);
}
#endif