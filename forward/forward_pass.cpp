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
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cublas_v2.h>
#include <cudnn.h>


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "operations.h"


using namespace cv;
using namespace std;

// #ifndef checkCUDNN
// #define checkCUDNN(expression)                               \
//   {                                                          \
//     cudnnStatus_t status = (expression);                     \
//     if (status != CUDNN_STATUS_SUCCESS) {                    \
//       std::cerr << "Error on line " << __LINE__ << ": "      \
//                 << cudnnGetErrorString(status) << std::endl; \
//       std::exit(EXIT_FAILURE);                               \
//     }                                                        \
//   }
// #endif

Mat loadImage(const char* image_path){
	Mat image = cv::imread(image_path, CV_LOAD_IMAGE_COLOR);
	image.convertTo(image, CV_32FC3);
	normalize(image, image, 0, 1, cv::NORM_MINMAX);
	return image;
}

float* Conv2D_func(cudnnHandle_t cudnn, Mat image)
{
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

	Conv2D conv1(3, 3, 3, 3, 1, 1, 1, 1, image.rows, image.cols, cudnn);
	conv1.SetWeights(&(h_kernel[0][0][0][0]));

	float* input = image.ptr<float>(0);

	float* h_output = conv1.ConvForward(input);
	
	return h_output;
}

float* MaxPool(cudnnHandle_t cudnn, Mat image)
{

	// Pool pool1(1,1,3,image.rows,image.cols,182 ,276,cudnn);
    Pool pool1(1,1,3,image.rows,image.cols,2,2,0,2,2,cudnn);
	float* input = image.ptr<float>(0);

	float* h_output = pool1.PoolForward(input);
	
	return h_output;

}

float* ReLU(cudnnHandle_t cudnn, float* image, int rows, int cols)
{

	Activation act1(1,1,3,rows,cols,cudnn);

	float* h_output = act1.ActivationForward(image);
	
	return h_output;

}

void save_image(const char* output_filename, float* buffer, int height, int width) {
		cv::Mat output_image(height, width, CV_32FC3, buffer);

		//Make negative values zero.
		// cv::threshold(output_image,
		// 				output_image,
		// 				/*threshold=*/0,
		// 				/*maxval=*/0,
		// 				cv::THRESH_TOZERO);
		namedWindow("image", WINDOW_AUTOSIZE);
		imshow("image", output_image);
		waitKey(0);
		// cv::normalize(output_image, output_image, 0.0, 255.0, cv::NORM_MINMAX);
		output_image.convertTo(output_image, CV_8UC3);

		cv::imwrite(output_filename, output_image);
}




int main(){

		cudnnHandle_t cudnn;
		checkCUDNN(cudnnCreate(&cudnn));

		//Read Image
		Mat image = loadImage("./data/n02118333_27_fox.jpg");
        float* h_output;
		h_output = Conv2D_func(cudnn,image);

		save_image("cudnnout.png", h_output, image.rows, image.cols);
        
        h_output = ReLU(cudnn,h_output,image.rows,image.cols);
        save_image("cudnnact.png", h_output, image.rows, image.cols);
		return 0;
		
}







