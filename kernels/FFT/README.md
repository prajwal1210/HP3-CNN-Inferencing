## FFT based convolution
#### Input pre-processing:
* Input is of the form `Batch_size * input_channels * H * W`
* **Pad the input:** Each Input layer is padded by the required number parallely using cuda kernel pad_input
* **Convert the input to frequency domain by FFT:** FFT of the input is obtained using CUFFT library’s functions 
  * **cufftPlanMany** that creates a plan supporting batched input
  * **cufftExecR2C** is used for real-to-complex forward transform for single precision.

#### Filter pre-processing:
* Each filter is of the form `output_channels * H * W` 
* **Flip around the center:** Each element of the filter is flipped around the center. Not doing this operation on the kernel results in Correlation (not convolution).
				
                                   F(i,j,k) = (D-i, H-j, W-k)

* **Pad the filter:** The filter is padded to the size of the input. This is essential in order to get the correct output by pointwise product in the frequency domain  
* **Align the filter:** The filter is required to have it’s Central element of the kernel in the (0,0) position. This is essential in order to get the correct output by pointwise product in the frequency domain

                            F(i,j,k) = ((i - D/2) % D, (j - H/2) % H, (k - W/2)% W)

* **Convert the filter to frequency domain by FFT:** FFT of the filter is obtained by CUFFT library’s functions 
  * **cufftPlanMany** that creates a plan supporting batched input
  * **cufftExecR2C** is used for real-to-complex forward transform for single precision

#### Convolution Operation:
* Preprocessed input and preprocessed filter in frequency domain are multiplied pointwise
* **Convert back the obtain product using Inverse FFT:** The final convolution result is obtained by CUFFT library’s functions 
  * **cufftPlanMany** that creates a plan supporting batched input
  * **cufftExecC2R** is used for complex-to-real forward transform for single precision
* This operation “performs” cyclic convolution: The convolution kernel wraps around the input borders in all the dimensions. But, we require the output to be clamped in the borders and hence require post-processing on the output.
* **High speed version:** The operation is looped over batch size and the precomputed FFT of input is replicated number of filters times and pointwise product is carried out.  This has a memory read overhead.
* **Low memory version:** The FFTs are calculated when required. It is looped over output channels. This leads to FFTs being calculated for the same input multiple times and hence is slow. This uses less memory though.

#### Output Post-Processing:
* **Crop and stride:** The output obtained in the previous step is cropped to `Input_size - filter_size + 1`  According to the input stride, the required elements are transferred to the output.

## Comparison with cuDNN’s FFT  
#### Shortcomings
* In the current implementation, there is loop on the `batch_size`, alleviating this again requires more memory. This makes the implementation slower
* In the CUDNN implementation, the input and filter are padded to powers of 2,3,5,7 and the performance and precision are best. Our implementation doesn't do this as this requires more memory (greater than google colab's limit). This makes our implementation slower.
* The current implementation is for square images. This can be easily scaled up to images with unequal height and width.

#### Improvements
* Input’s feature map need not have `height + 2 * zero-padding` <= 256
* Input’s feature map need not have `width + 2 * zero-padding`  <= 256
* The vertical and horizontal filter stride need not be equal to 1
* Filter’s height, width need not be greater than zero-padding height, width

## References
* http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.472.2396&rep=rep1&type=pdf
* https://docs.nvidia.com/cuda/cufft/index.html#fftw-conversion-guide
