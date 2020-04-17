/************************************************************
 * data_util.cc:								                            *
 * Implementation of data_util.h library                    *
 *                                                          *
 * Author: Prajwal Singhania								                *
 ************************************************************/

#include "data_util.h"

/* Implementation of (ImageLoader)loadImage */
cv::Mat ImageLoader::loadImage(const char* image_path, bool resize) {
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

/* Implementation of (ImageLoader)HWCToCHW */
float* ImageLoader::HWCToCHW(float* image, int rows, int cols, int channels) {

	float* data = (float*)malloc(channels*rows*cols*sizeof(float));
	for(size_t i=0; i<rows; i++) {
    for(size_t j=0; j<cols; j++) {
      for(size_t c=0; c<channels; c++) {
          data[c*rows*cols + i*cols + j] = (float) image[i*cols*channels + j*channels + c];
      }
    }
	}
	return data;
}

/* Implementation of (ImageLoader)CHWToHWC */
float* ImageLoader::CHWToHWC(float* image, int rows, int cols, int channels) {

	float* data = (float*)malloc(channels*rows*cols*sizeof(float));
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
  image.release();
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


/* Implementation of MiniImageNetLoader constructor */
MiniImageNetLoader::MiniImageNetLoader(const char* datapath, int batchsize) {
  this->dataPath = datapath;
  this->BATCH_SIZE = batchsize;
  std::string imageListPath = this->dataPath + "ImageLists.txt";
  std::cout << imageListPath << std::endl;
  this->imageListFile.open(imageListPath.c_str(), std::ifstream::in);
  if(!this->imageListFile.is_open()) {
    std::cerr << "Error opening the dataset description file" << std::endl;
    exit(1);
  }
}

/* Implementation of MiniImageNetLoader destructor */
MiniImageNetLoader::~MiniImageNetLoader() {
  this->closeFile();
}

/* Implementation of (MiniImageNetLoader)loadNextBatch */
float* MiniImageNetLoader::loadNextBatch(int& batchsize, int& channels, int& height, int& width, bool& completed) {
  std::string fileName;
  int batch_cnt = 0;
  float* data = (float*)malloc(this->BATCH_SIZE * 3 * 256 * 256 * sizeof(float));
  int data_ptr = 0;
  while(std::getline(this->imageListFile, fileName)) {
    batch_cnt++;
    std::string imagePath = this->dataPath + fileName;
    cv::Mat image = this->loadImage(imagePath.c_str(), true);
    float* im = image.ptr<float>(0);
    float* im_chw = this->HWCToCHW(im, image.rows, image.cols, 3);
    for(int k = 0; k < 3 * image.rows * image.cols; k++) {
      data[data_ptr++] = im_chw[k];
    }
    image.release();
    free(im_chw);
    if(batch_cnt == this->BATCH_SIZE)
      break;
  }
  height = 256;
  width = 256;
  batchsize = batch_cnt;
  channels = 3;
  if(batch_cnt < this->BATCH_SIZE) {
    completed = true;
    this->closeFile();
  }
  return data;
}

/* Implementation of (MiniImageNetLoader)closeFile */
void MiniImageNetLoader::closeFile() {
  if(this->imageListFile.is_open()) {
    this->imageListFile.close();
  }
}