# Introduction
This folder contains the main libraries for the CNN Inferencing Engine, the Data Utilities, and the Correctness Tests

# Inferencing Engine Libraries
We provide two C++ libraries to help run the forward pass of a CNN by loading a model saved in our custom specification format:

* `operations.h` 
	This contains classes for the major operations of a CNN which include:
	* Conv2D : Convolution Opertation [The convolution algorithm can be selected with the help of a parameter]
	* Pool (Pool2D) : Provides option for both MAX and AVERAGE pooling as well as support for Adaptive Pooling:
		* Adaptive Pooling is when you specify only the output size and the operation itself computes the filter sizes etc. to achieve that given any input. Thi s was added to support Pytorch's Adaptive Pooling Layer. However, the algorithm it uses is different from Pytorch and so the answers can differ. 
	* Activation : Provides both RELU and SIGMOID activations
	* Linear : Provides support for Linear/FC computations using GEMM
* `cnn_forward.h`
This libary provides two functions:
	* `loadCNNModelFromFile`: Helps load a saved model into a Network object
	* `forwardPass`: Iterates over the CNN Network object and computes the forward pass on the provided data

# Data Utilities
The `data_util.h` library provides classes to load a single image as well as our custom Mini Image Net Dataset in batches:
* *Mini Image Net Dataset:*
	We created this dataset by downloading a portion of the ImageNet using the tool - [ImageNet Downloader Tool](https://github.com/mf1024/ImageNet-Datasets-Downloader) and then resizing the images to 256x256. The dataset contains 372 images

# Correctness Tests
We have 3 tests that extensively check the correctness of differenet functionalities of the inferencing engine. The tests are located in different folders in the this directory. The tests are:

 1. **Test-1 Operations Test**
	> Located under opearations_test/

	This tests the individual component operations in the operations.h library and compares the results with Pytorch output
	
 2. **Test-2 Single Image CNN Forward Test**
	> Located under cnn_forward_test/
		
	This tests the CNN Forward library by running both VGG19 and Alexnet on a single image and comparing the results with Pytorch output
	
 3. **Test-3 Batch Image CNN Forward Test**
	> Located under batch_test/
	
	This tests the CNN Forward library for a batch of 8 images by running both VGG19 and Alexnet and comparing the results with Pytorch output