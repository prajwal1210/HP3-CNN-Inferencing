# Introduction
This tests the individual component operations in the operations.h library and compares the results with Pytorch Output

# How to run the test
 * **Compilation:**   `$ make test`  [Can be skipped if compiled through the global makefile]
 * **Running the tests:**
	```
	$ make run_direct		# For Direct Convolution
	$ make run_fft			# For FFT Convolution
	$ make run_winograd		# For Winograd Convolution
	$ make run_im2col		# For IM2COL & GEMM Convolution
    $ make run_cudnn		# For CUDNN API Convolution
	```
 * **Clean the test directory:**  ```$ make clean```
