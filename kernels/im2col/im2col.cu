#include "im2col.hpp"


// converts an image of shape: ic x ih x iw (ic: input_channels in image)
// to 2D col of shape: (ic * kh * kw) x (hcol * wcol)
// data_im is pointer to image, data_col is pointer to output
__global__ void im2col_kernel(const float * data_im, float * data_col, const int n,
							  const int kh, const int kw, const int pad, const int stride,
							  const int ih, const int iw, const int ic,
							  const int hcol, const int wcol) 
{
	// esentially this loop would have run batch size number of times
	// but since we are iterating over each image separately, it executes just once
	CUDA_KERNEL_LOOP(index, n) {
		int w_out = index % wcol;
		index /= wcol;
		int h_out = index % hcol;
		int channel_in = index / hcol;
		int channel_out = channel_in * kh * kw;
		int h_in = h_out * stride - pad;
		int w_in = w_out * stride - pad;
		data_im += (channel_in * ih + h_in) * iw + w_in;
		data_col += (channel_out * hcol + h_out) * wcol + w_out;
		#pragma unroll
		for (int i = 0; i < kh; ++i) {
			for (int j = 0; j < kw; ++j) {
				int h = h_in + i;
				int w = w_in + j;
				*data_col = (h >= 0 && w >= 0 && h < ih && w < iw) ?
				  0: data_im[i * iw + j];
				data_col += hcol * wcol;
			}
		}
	}
}

// takes in the image on GPU: ic x ih x iw (ic: input channels)
// and the kernels on GPU: oc x ic x kh x kw (oc: output channels)
// does the convolution and returns
void im2col_gemm_gpu(const float * data_im, const float * data_ker, cublasHandle_t handle,
					 const int kh, const int kw, const int pad, const int stride,
					 const int ih, const int iw, const int ic, const int oc,
					 float * data_col, float * data_out)
{
	// Step 1: convert the image to col form
	
	// dimensions of the col corr to this image
	int hcol = (ih + 2 * pad - kh) / stride + 1;
	int wcol = (iw + 2 * pad - kw) / stride + 1;

	float* im_ptr = (float *)malloc(ic * ih * iw * sizeof(float));
	CUDA_CHECK(cudaMemcpy(im2ptr, data_im, ic*ih*iw*sizeof(float), cudaMemcpyDeviceToHost));
	for(int l = 0; l < ic; l++)
	{
	  for(int i = 0; i < ih; i++)
	  {
		for(int j = 0; j < iw; j++)
		  std::cout << im_ptr[l * ih * iw + i * iw + j] << " ";	
		printf("\n");
	  }
	  printf("\n\n");
	}
	free(im_ptr);

	// We are going to launch ic * hcol * wcol kernels threads for im2col,
	// each thread is responsible for copying a single-channel grid
	// one thread per output pixel in the output of conv
	int op_size = ic * hcol * wcol;
	im2col_kernel<<<GET_BLOCKS(op_size), CUDA_NUM_THREADS>>>(
		data_im, data_col, op_size, kh, kw, pad, stride, ih, iw, ic, hcol, wcol);
	CUDA_POST_KERNEL_CHECK; // check if there was any error

	float* im2col_ptr = (float *)malloc(ic * kh * kw * hcol * wcol * sizeof(float));
	CUDA_CHECK(cudaMemcpy(im2col_ptr, data_col, ic*kh*kw*hcol*wcol*sizeof(float), cudaMemcpyDeviceToHost));
	for(int l = 0; l < oc; l++)
	{
	  for(int i = 0; i < hcol; i++)
	  {
		for(int j = 0; j < wcol; j++)
		{
		  for (int k = 0; k < kh * kw; ++k)
		  	std::cout << hcol * wcol * l * kh * kw + i * wcol + j + k * hcol * wcol << ": ";
			std::cout << im2col_ptr[hcol * wcol * l * kh * kw + i * wcol + j + k * hcol * wcol] << " ";
		  printf("\n");
		}	
		printf("\n");
	  }
	  printf("\n");
	}
	free(im2col_ptr);
	std::exit(EXIT_SUCCESS);

	// now, the col form shall be multiplied with the kernels laid out straight i.e. (ic * kh * kw)
	// so, since, oc is the number of kernels, we get:
	// "2D kernel matrix" oc x (ic * kh * kw)
	// and the "2D col matrix" is: (ic * kh * kw) x (hcol * wcol)
	// and you see that magically, their multiplication output is:
	// output: oc x (hcol * wcol)... ie oc x hcol x wcol, the exact shape needed by next im2col
	// so, there is no need to ever work things back or reshape either
	// in sumamary, we do matmul(kernel flatten, im2col(im_input)) -> conv_output (in "correct" form)

	// Step 2: GEMM using libcublas

	
	// get params ready for GEMM call
	const float alpha = 1.0f;
	const float beta  = 0.0f;
	int ldA, ldB, ldC;
	int m = ldA = ldC = hcol * wcol;
	int n = ldB = oc;
	int k = ic * kh * kw;
	
	// CUDA sees matrices as column major
	// So, a matrix we see as HxW, it would see as WxH in the same memory layout
	// So, matA (our view) -> matA' (CUDA view)
	// Thus, to do matA * matB in our view, we shall run CUDA for matB * matA.
	// Output would be matB' * matA' (CUDA view) = (matA * matB)' (CUDA view) = matA * matB (our view)
	// In essence, trust me when I do col * kernel to achieve kernel * col
	cublasStatus_t ret =
		cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha,
					data_col, ldA, data_ker, ldB, &beta, data_out, ldC);
	CUBLAS_CHECK(ret, "cublas Sgemm returned an error!");
}

float * im2colWithCuda(const float * dev_image, const float * dev_kernel, const int batch,
					   const int kh, const int kw, const int pad, const int stride,
					   const int ih, const int iw, const int ic, const int oc)
{
	// image dim
	ssize_t image_size = ic * ih * iw;
	ssize_t images_size = batch * image_size;

	// kernel dim
	ssize_t K = ic * kh * kw;
	ssize_t kernels_size = oc * K;

	// col dim
	ssize_t hcol = (ih + 2 * pad - kh) / stride + 1;
	ssize_t wcol = (iw + 2 * pad - kw) / stride + 1;
	ssize_t one_col = ic * kh * kw * hcol * wcol;
	ssize_t col_batch = batch * one_col;

	// output dim
	ssize_t output_feature = oc * hcol * wcol;	
	ssize_t result_size = batch * output_feature;
	
	// // image
	// float * dev_image = nullptr;
	// CUDA_CHECK(cudaMalloc((void**)&dev_image, images_size * sizeof(float)));
	// CUDA_CHECK(cudaMemcpy(dev_image, data_im, images_size * sizeof(float), cudaMemcpyHostToDevice));
	
	// // kernel
	// float * dev_kernel = nullptr;
	// CUDA_CHECK(cudaMalloc((void**)&dev_kernel, kernels_size * sizeof(float)));
	// CUDA_CHECK(cudaMemcpy(dev_kernel, data_ker, kernels_size * sizeof(float), cudaMemcpyHostToDevice));

	// col
	float * dev_col = nullptr;
	CUDA_CHECK(cudaMalloc((void**)&dev_col, col_batch * sizeof(float)));

	// result
	float * dev_ret = nullptr;
	CUDA_CHECK(cudaMalloc((void**)&dev_ret, result_size * sizeof(float)));

	// cuBLAS initialize
	cublasHandle_t handle;
	CUBLAS_CHECK(cublasCreate(&handle), "cublasCreate() error!");

	// loop over the batch
	const float * t_dev_image = dev_image;
	float * t_dev_col = dev_col;
	float * t_dev_ret = dev_ret;
	for(int i = 0; i < batch; i++)
	{
		// Launch GPU kernel to work on each image
		im2col_gemm_gpu(t_dev_image, dev_kernel, handle, kh, kw, pad, 
							stride, ih, iw, ic, oc, t_dev_col, t_dev_ret);

		t_dev_image += image_size;
		t_dev_col += one_col;
		t_dev_ret += output_feature;
	}

	// cuBLAS finalize
	CUBLAS_CHECK(cublasDestroy(handle), "cublasDestroy() error!");
	
	// Check for any errors launching the kernel
	CUDA_CHECK(cudaGetLastError());
	
	// // cudaDeviceSynchronize waits for the kernel to finish, and returns
	// // any errors encountered during the launch.
	// cudaStatus = cudaDeviceSynchronize();
	// if (cudaStatus != cudaSuccess) {
	// 	fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching im2col 
	// Kernel!\n", cudaStatus);
	// 	goto Error;
	// }

	// sdkStopTimer(&timer);
	// double elapsedTimeInMs = sdkGetTimerValue(&timer);
	// printf("caffe is %fms\n", elapsedTimeInMs);

	// Copy output vector from GPU buffer to host memory.
	// CUDA_CHECK(cudaMemcpy(data_col, dev_col, N * K *batch_size* sizeof(float), cudaMemcpyDeviceToHost));
	float * data_ret = (float *)malloc(result_size * sizeof(float));
	CUDA_CHECK(cudaMemcpy(data_ret, dev_ret, result_size * sizeof(float), cudaMemcpyDeviceToHost));

	// cudaFree(dev_image);
	cudaFree(dev_col);
	// cudaFree(dev_kernel);
	cudaFree(dev_ret);
	
	// sdkDeleteTimer(&timer);

	return data_ret;
}

float* IM2COL::forward(int out_size, int channel, int kernel_height, int kernel_width, int pad, 
		int stride, float* kernel, int batch_size, int input_height, int input_width, float* input)
{
	return im2colWithCuda(input, kernel, batch_size, kernel_height, kernel_width, 
					pad, stride, input_height, input_width, channel, out_size);
}


