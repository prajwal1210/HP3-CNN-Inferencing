#ifndef DATA_UTIL_H 
#define DATA_UTIL_H

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <cfloat>

#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

class SingleImageLoader {
 public:
  /* Loads the image in colored CHW format at the specified path and returns a pointer to it
  * Input  : Path to the image
  * Output : float pointer to the image (side effects - sets the dimensions of the image)
  */
  float* loadSingleColoredImageCHW(const char* image_path, int& batchsize, int& channels, int& height, int& width, bool resize = true);

  /* Saves the image at the specified path after converting to CHW, thresholding, normalising and converting to 8-bit format
  * Input  : Output File Path, Image Buffer, Height of Image, Width of Image
  */
  void saveImageCHW(const char* output_filename, float* buffer, int channels, int height, int width);

 private:
  /* Loads the image in colored format at the specified path into a cv::Mat object, converts it to float and normalizes the value to 0 and 1
  * Input  : Path to the image
  * Output : cv::Mat object 
  */
  cv::Mat loadImage(const char* image_path, bool resize);

  /* Converts and image in HWC format to CHW format
  * Input  : Image Buffer in HWC format, Rows of Image, Cols of Image, Channels of the Image
  * Output : An Image buffer in CWH format
  */
  float* HWCToCHW(float* image, int rows, int cols, int channels);

  /* Converts and image in CHW format to HWC format
  * Input  : Image Buffer in CHW format, Rows of Image, Cols of Image, Channels of the Image
  * Output : An Image buffer in HWC format
  */
  float* CHWToHWC(float* image, int rows, int cols, int channels);
};

#endif