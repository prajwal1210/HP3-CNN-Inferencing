## Function Protoptype for Kernel Implementation

	forward(int out_channels, int input_channels, int kernel_height, int kernel_width, int padding, int stride, float* kernel_weights
			int batchsize_of_data, int input_height, int input_width, float* input)
Here :
* out_channels  = number of filters in the layer
* weights, input : pointer to the device (CUDA) arrays storing the necessary values. Both single-D arrays will be in NCHW format ( in 3D, the fastest changing dimension will be W, then H, then C )
* Padding, Stride : Assume same in both directions
* Kernel_height, kernel_width : Ideally, they can be different, but if it is it too hard to implement, you can assume that the input will always have kernel_height = kernel_width 

## TODO
* Complete all basic kernel implementations **(Rough deadline : Saturday - 4th April)**
* Test all the kernels individually **(Rough deadline: Sunday - 5th April)**
* Improve upon the Kernels if required (ex, Winograd for more than 3/5 filter size) 
* Integrating Kernels with the Forward Pass and Testing **(Hard Deadline: Wednesday - 8th April)**
* Adding Profilining, Plotting and Analysis for both VGG and AlexNet **(Hard Deadline: Saturday - 11th April)**
* Presentation and Documentation **(Hard Deadline: Monday - 13th April )**

## Compile Instructions

### Tests
All tests are located in folders with _test at the end and are run using:
* Compile - `make`
* Run the test - `make run` 
* Clean the test directory - `make clean`
* Requirements - OpenCV, Pytorch10.1

#### CNN Forward Test 
Located under forward/cnn_forward_test:
* Tests the CNN Forward library by running both VGG19 and Alexnet on a single image and comparing the results with Pytorch Output

#### Batch Test 
Located under forward/batch_test:
* Tests the CNN Forward library for a batch of 8 images by running both VGG19 and Alexnet and comparing the results with Pytorch Output

#### Operations Test 
Located under forward/opearations_test:
* Tests the individual component operations in the operations.h library and compares the results with Pytorch Output

## Using MiniImageNet Data
We have created a custom dataset from ImageNet with 372 images and resized them to 256X256. To use the dataset and run the above tests, make sure you extract the zip - MiniImageNet.zip present in forward/data folder first 