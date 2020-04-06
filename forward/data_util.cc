#include "data_util.h"

/* Implementation of (SingleImageLoader)loadImage */
cv::Mat SingleImageLoader::loadImage(const char* image_path, bool resize) {
	cv::Mat image = cv::imread(image_path, CV_LOAD_IMAGE_COLOR);
	image.convertTo(image, CV_32FC3);
  if (resize) {
    cv::Size size(256,256);
    cv::Mat resized_image;
    cv::resize(image,resized_image,size);//resize image
    cv::normalize(resized_image, resized_image, 0, 1, cv::NORM_MINMAX);
	  return resized_image;
  }
  else {
	  cv::normalize(image, image, 0, 1, cv::NORM_MINMAX);
	  return image;
  }
}

/* Implementation of (SingleImageLoader)HWCToCHW */
float* SingleImageLoader::HWCToCHW(float* image, int rows, int cols, int channels) {

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

/* Implementation of (SingleImageLoader)CHWToHWC */
float* SingleImageLoader::CHWToHWC(float* image, int rows, int cols, int channels) {

	float* data = new float[channels*rows*cols];
	for(size_t i=0; i<rows; i++) {
    for(size_t j=0; j<cols; j++) {
      for(size_t c=0; c<channels; c++) {
          data[i*cols*channels + j*channels + c] = (float) image[c*rows*cols + i*cols + j];
      }
    }
	}
	return data;
}

/* Implementation of (SingleImageLoader)loadSingleColoredImageCHW */
float* SingleImageLoader::loadSingleColoredImageCHW(const char* image_path, int& batchsize, int& channels, int& height, int& width, bool resize) {
  cv::Mat image = loadImage(image_path, resize);
  float* im = image.ptr<float>(0);
  height = image.rows;
  width = image.cols;
  batchsize = 1;
  channels = 3;
  float* retImage = this->HWCToCHW(im, height, width, channels);
  return retImage;
}

/* Implementation of (SingleImageLoader)saveImageCHW */
void SingleImageLoader::saveImageCHW(const char* output_filename, float* buffer, int channels, int height, int width) {
  buffer = this->CHWToHWC(buffer, height, width, channels);
  cv::Mat image(height, width, CV_32FC3, buffer);
  cv::Mat output_image = image.clone();
  /*namedWindow("image", WINDOW_AUTOSIZE);
  imshow("image", output_image);
  waitKey(0);*/
  cv::threshold(output_image,
          output_image,
          /*threshold=*/0,
          /*maxval=*/0,
          cv::THRESH_TOZERO);
  cv::normalize(output_image, output_image, 0.0, 255.0, cv::NORM_MINMAX);
  output_image.convertTo(output_image, CV_8UC3);
  cv::imwrite(output_filename, output_image);
}

