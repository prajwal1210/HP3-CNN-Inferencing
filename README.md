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
* Forward pass using the saved specification and the implemented library - Includes loading data as well (will take time) **(Deadline: Sunday - 5th April)**
* Integrating Kernels with the Forward Pass and Testing **(Hard Deadline: Wednesday - 8th April)**
* Adding Profilining, Plotting and Analysis for both VGG and AlexNet **(Hard Deadline: Saturday - 11th April)**
* Presentation and Documentation **(Hard Deadline: Monday - 13th April )**

## Compile Instructions

forward.cc:
	nvcc -I . -L /usr/local/cuda-10.1/man/man7/libcublas.so.7 -std=c++14 forward/forward.cc proto/translator.cpp proto/network.pb.cc forward/operations.cpp -lcudnn -lcublas `pkg-config --cflags --libs protobuf`