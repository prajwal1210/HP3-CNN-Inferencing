## Testing your kernel outputs

## Compile Instructions

### Compiling Proto and Downloading the Pretrained Models and save them in protobuf format

```
$ cd proto
$ protoc -I=. --cpp_out=. ./network.proto
$ cd ..
$ python ConvertToSpecification.py
```

### Test-1 Operations Test 
Located under forward/opearations_test:
* Tests the individual component operations in the operations.h library and compares the results with Pytorch Output
* Compile - `make test`
* Run the test for direct convolution - `make run_direct` 
* Run the test for FFT-based convolution - `make run_fft`
* Run the test for Winograd-based convolution - `make run_winograd` 
* Run the test for im2col-based convolution - `make run_im2col`
* Clean the test directory - `make clean`
* Requirements - OpenCV, Pytorch10.1

### Setup Protobuf
```shell
$ apt-get install autoconf automake libtool curl make g++ unzip
$ git clone https://github.com/protocolbuffers/protobuf.git
$ cd protobuf
$ git submodule update --init --recursive
$ ./autogen.sh
$ ./configure
$ make -j8
$ make check
$ sudo make install
$ sudo ldconfig 
$ cd ..
```

#### Test-2 CNN Forward Test 
Located under forward/cnn_forward_test:
* Tests the CNN Forward library by running both VGG19 and Alexnet on a single image and comparing the results with Pytorch Output
* Compile - `make test`
* Run the test for direct convolution - `make run_direct` 
* Run the test for FFT-based convolution - `make run_fft`
* Run the test for Winograd-based convolution - `make run_winograd` 
* Run the test for im2col-based convolution - `make run_im2col`
* Clean the test directory - `make clean`
* Requirements - 

## Using MiniImageNet Data
We have created a custom dataset from ImageNet with 372 images and resized them to 256X256. To use the dataset and run the batch tests, extract the zip - MiniImageNet.zip present in forward/data folder first

```
$ cd forward/data/MiniImageNet.zip
$ unzip MiniImageNet.zip
$ cd ..
```

#### Test-3 Batch Test 
Located under forward/batch_test:
* Tests the CNN Forward library for a batch of 8 images by running both VGG19 and Alexnet and comparing the results with Pytorch Output
* Compile - `make test`
* Run the test for direct convolution - `make run_direct` 
* Run the test for FFT-based convolution - `make run_fft`
* Run the test for Winograd-based convolution - `make run_winograd` 
* Run the test for im2col-based convolution - `make run_im2col`
* Clean the test directory - `make clean`
* Requirements - 

##### Notebook run the tests on Google Colab - https://colab.research.google.com/drive/1GD7mgy3pVIKSnobhY7hSdD_3P9i5532R
