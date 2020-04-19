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
#include <vector>


#include "forward/operations.h"
#include "proto/network.pb.h"
#include "proto/translator.h"

namespace CNN {
  /* Loads the CNN Model from the specification file into the provided object reference
   * Input  : Path to the model, Reference to DeepNet::Network object
   */
  void loadCNNModelFromFile(const char* model_file, DeepNet::Network& net);
  
  /* Iterates over the CNN Network and computes the forward pass on the provided data
   * Input  : DeepNet::Network object, references to dimensions of the input, pointer to the input array, Algorithm to use for Convolution,
   *          Reference to a vector to store the elapsed time details, Reference to the status flag
   * Output : Pointer to the final layer output (side effect - sets the dimensions accordingly, add the elapsed time details for convolution in the vector)
   */
  float* forwardPass(DeepNet::Network net, int& batchsize, int& input_h, int& input_w, int& input_c, float* input, 
                    customAlgorithmType algo, std::vector<profilingElapsedTime>& time_elapsed, bool& succes);
}
#endif