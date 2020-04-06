/************************************************************
 * forward_pass_test.cpp:									                  *
 * This is the C++ script to test the operations.h library 	*
 * 															                            *
 * Author: Prajwal Singhania								                *
 ************************************************************/
 
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <cfloat>

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>
#include <cudnn.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "forward/operations.h"

using namespace cv;
using namespace std;



/* Loads the image in colored format at the specified path into a cv::Mat object, converts it to float and normalizes the value to 0 and 1
 * Input  : Path to the image
 * Output : cv::Mat object 
 */
Mat loadImage(const char* image_path) {
	Mat image = cv::imread(image_path, CV_LOAD_IMAGE_COLOR);
	image.convertTo(image, CV_32FC3);
	normalize(image, image, 0, 1, cv::NORM_MINMAX);
	return image;
}

/* Saves the image at the specified path after thresholding, normalising and converting to 8-bit format
 * Input  : Output File Path, Image Buffer, Height of Image, Width of Image
 */
void save_image(const char* output_filename, float* buffer, int height, int width) {
  cv::Mat image(height, width, CV_32FC3, buffer);
  cv::Mat output_image = image.clone();
  // namedWindow("image", WINDOW_AUTOSIZE);
  // imshow("image", output_image);
  // waitKey(0);

  cv::threshold(output_image,
          output_image,
          /*threshold=*/0,
          /*maxval=*/0,
          cv::THRESH_TOZERO);
  cv::normalize(output_image, output_image, 0.0, 255.0, cv::NORM_MINMAX);
  output_image.convertTo(output_image, CV_8UC3);

  cv::imwrite(output_filename, output_image);
}


/* Converts and image in HWC format to CHW format
 * Input  : Image Buffer in HWC format, Rows of Image, Cols of Image, Channels of the Image
 * Output : An Image buffer in CWH format
 */
float* HWCToCHW(float* image, int rows, int cols, int channels) {

	float* data = new float[channels*rows*cols];
	for(size_t i=0; i<rows; i++) {
    for(size_t j=0; j<cols; j++) {
      for(size_t c=0; c<channels; c++) {
          data[c*rows*cols + i*cols + j] = (float) image[i*cols*channels + j*channels + c];
      }
    }
	}
	return data;
}

/* Converts and image in CHW format to HWC format
 * Input  : Image Buffer in CHW format, Rows of Image, Cols of Image, Channels of the Image
 * Output : An Image buffer in HWC format
 */
float* CHWToHWC(float* image, int rows, int cols, int channels) {

	float* data = new float[channels*rows*cols];
	for(size_t i=0; i<rows; i++) {
    for(size_t j=0; j<cols; j++) {
      for(size_t c=0; c<channels; c++) {
          data[i*cols*channels + j*channels+ c] = (float) image[c*rows*cols + i*cols + j];
      }
    }
	}
	return data;
}	

float* Conv2D_func(cudnnHandle_t cudnn, float* input, int in_h, int in_w, int& out_c, int& out_h, int& out_w) {
	const float kernel_template[3][3] = {
	{1,  1, 1},
	{1, -8, 1},
	{1,  1, 1}
	};

	float h_kernel[3][3][3][3];
	for (int kernel = 0; kernel < 3; ++kernel) {
	  for (int channel = 0; channel < 3; ++channel) {
			for (int row = 0; row < 3; ++row) {
			  for (int column = 0; column < 3; ++column) {
					h_kernel[kernel][channel][row][column] = kernel_template[row][column];
			  }
		  }   
	  }
	}

	Conv2D conv1(3, 3, 3, 3, 1, 1, 1, 1, in_h, in_w, cudnn);
	conv1.SetWeights(&(h_kernel[0][0][0][0]));

	float* h_output = conv1.ConvForward(input);

	int out_n, c, h, w;
	conv1.GetOutputDims(&out_n, &c, &h, &w);
	out_c = c;
	out_h = h;
	out_w = w;
	
	return h_output;
}

float* MaxPool_func(cudnnHandle_t cudnn, float* image, int rows, int cols, int& out_h, int& out_w) {

    Pool pool1(1,1,3,rows,cols,4,4,0,4,4,cudnn);

	float* h_output = pool1.PoolForward(image);

	int out_n, out_c, h, w;
	pool1.GetOutputDims(&out_n, &out_c, &h, &w);
	
	out_h = h;
	out_w = w;

	return h_output;

}

float* Signmoid_func(cudnnHandle_t cudnn, float* image, int rows, int cols, int& out_h, int& out_w) {

	Activation act1(1,1,3,rows,cols,cudnn);

	float* h_output = act1.ActivationForward(image);

	int out_n, out_c, h, w;
	act1.GetOutputDims(&out_n, &out_c, &h, &w);

	out_h = h;
	out_w = w;
	
	return h_output;

}

float* getValues(int size) {
	int MAX = 1024;
	float* w = new float[size];
	for(int i = 0; i < size; i++)
		w[i] = i%MAX;
	return w;
}

float* Linear_func(cudnnHandle_t cudnn, float* image, int out_nodes, int in_nodes) {

	cublasHandle_t cublasHandle;
	checkCudaErrors(cublasCreate(&cublasHandle));

	Linear lin1(1, out_nodes, in_nodes, cublasHandle);


	float* Weight = getValues(out_nodes*in_nodes);
	float* Bias = getValues(out_nodes);

	lin1.SetWeights(Weight);
	lin1.SetBias(Bias, cudnn); 

	float* h_output = lin1.LinearForward(image);

	return h_output;
}


int main() {
  cudnnHandle_t cudnn;
  checkCUDNN(cudnnCreate(&cudnn));

  /* Read Image */
  Mat image = loadImage("../data/n02118333_27_fox.jpg");
      float* im = image.ptr<float>(0);
  float* re_im;

  int batchsize = 1;
  int channels = 3;
  int h = image.rows;
  int w = image.cols;
  
  im = HWCToCHW(im, h, w, channels);

  //Reconvert and check
  re_im = CHWToHWC(im, h, w, channels);
  save_image("reconvert.png", re_im, h, w);

  /* Conv Layer */
  float* h_output;
  h_output = Conv2D_func(cudnn, im, h, w, channels, h, w);
  re_im = CHWToHWC(h_output, h, w, channels);
  save_image("cudnnconv.png", re_im, h, w);

  /* Pool Layer */
  h_output = MaxPool_func(cudnn, h_output, h, w, h, w);
  re_im = CHWToHWC(h_output, h, w, channels);
  save_image("cudnnpool.png", re_im, h, w);

  /* Activation Layer */
      h_output = Signmoid_func(cudnn,h_output, h, w, h, w);
  re_im = CHWToHWC(h_output, h, w, channels);
      save_image("cudnnact.png", re_im, h, w);

  FILE* fp;
  fp = fopen("linear.txt" , "w");

  /* Linear Layer */
  h_output = Linear_func(cudnn, h_output, 100, batchsize*channels*h*w );
  for(int i=0;i<100;i++)
    fprintf(fp, "%.1f ",h_output[i]);
  fprintf(fp, "\n");


  /* Only Linear Layer Test */
  float* input = getValues(100);
  float* output = Linear_func(cudnn, input, 10, 100);
  for(int i=0;i<10;i++)
    fprintf(fp, "%.1f ",output[i]);
  fprintf(fp, "\n");

  fclose(fp);

  return 0;		
}
