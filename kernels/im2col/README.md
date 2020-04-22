## Im2Col based convolution

**Note: This README describes the implementation in im2col.cu. The differences in im2col_single.cu and im2col2.cu are highlighted in our presentation, and are reproduced here for reference:**
* im2col_single.cu: Naive implementation. Iterates over each image in batch. Uses normal GEMM.
* im2col2.cu: Faster implementation. A single kernel launch iterates over each image in batch, instead of multiple kernels launched (as in im2col_single). But number of threads for the launch is same as im2col_single. Uses batched GEMM.
* im2col.cu: Fastest implementation. A single kernel is launched which parallely processes each image in the batch, ie launching with batch size times the number of threads in im2col_single (and im2col2). Uses batched GEMM.
* Comparison for the three implementations is provided in the slides.
* More details on the implementation are present as comments within the codes.

### Im2Col: Kernel im2col_kernel
* Input is of the form `batch_size * input_channels * H * W`
* We convert each image to col form `batch_size * input_channels * hcol * wcol`
* **Input padding is done implicitly while tiling the image**

#### The Implementation:
1. We use the stride and pad information to figure out what all places the kernel will be multiplied with the image.
2. We launch one thread for each such point to create the kernel multiplication patch centred at that point.
3. That thread adds the padding as necessary when copying the image pixels.
4. We do this in parallel for each image channel.

**Note on kernel launch**
* For bs images of size ic x ih x iw, each leading to col form of size (ic * kh * kw) x (hcol * wcol):
  - We launch ic * hcol * wcol threads (so, total of bs * ic * hcol * wcol)
  - Each thread is, thus, responsible for copying its relevant kh * kw sized patch in col form
* Maximum possible threads for each block are launched to maximize GPU efficiency. Thus, the grid size is (bs, ceil(ic * hcol * wcol / MAX_THREADS)), and block size is MAX_THREADS

### GEMM: Kernel cublasSgemmStridedBatched
* Multiplies the col matrix: `bs x input_channels x hcol x wcol`
* With the kernel matrix: `output_channels x input_channels x kh x hw`
* To get the convolution output `batch_size x output_channels x hcol x wcol`
* **This version of GEMM does the matmul on all the images in the batch in parallel**

#### The Implementation:
1. Performs C + i\*strideC = α op(A + i\*strideA) op(B + i\*strideB) + β(C + i\*strideC) for each i ∈ [0, batchSize − 1]
2. Essentially, it does parallel matrix multiplications for a batch, where all input matrices are provided together sequentially, ie all matrices are some `stride` distance apart within their batch. Thus, this one call will parallely do the matrix multiplication for all images in the batch.
3. Strides:
  - strideA = size of each col (ic x kh x kw x hcol x wcol), 
  - strideB = 0 as same kernel matrix to be used for each multiplication
  - strideC = size of each output feature map (oc x hcol x wcol)
4. Since we are simply doing A * B, we need α = 1, β = 0 and since we don't need any transpose, op = CUBLAS_OP_N

**Note on memory layout**
* cublas considers matrices to be laid out in column major form.
* So, a matrix we see as HxW, cublas would see as WxH in the same memory layout. matA (HxW) [our view] = matA' (WxH) [CUDA view] \(' denotes transpose\)
* Thus, to do matA * matB in our view, we shall run CUDA for matB * matA. Here's how it works:
  - GEMM(matB, matA) [our view] = matB' * matA' [CUDA view] = (matA * matB)' [CUDA view] = matA * matB [our view]
  - Thus, by running GEMM on col * kernel, we actually achieve matmul(kernel * col) in row-major form. This sidesteps the need to do any transpose or column reshuffling and thus, improves performance.


## Performance improvements from naive implementation:
* Padding is done implicitly, while performing tiling. This saves a lot of overhead.
* Maximum possible threads for each block are launched to maximize GPU efficiency.
* Even though cuBLAS considers matrices to be column major, the matrix multiplication is done cleverly so it does not require any awkward dimension reshuffling.
* Instead of performing GEMM on each image separately, we used the batched version of cublas GEMM.
* Instead of serially launching im2col for each image in a batch, we process batches in parallel. We compared two approaches: (N = output feature map size for one image)
i. launching N threads that each iterate over the images in the batch.
ii. launching batch size x N threads, so each thread only has one image to take care of.
  - Both gave similar results, but option ii. was slightly faster than i.
  - Both options were quite faster than the serial approach. 2x faster in case of AlexNet.
  - Performance gain was much more for AlexNet than VGG as it has larger filters, so parallelization increases.
* Metrics for the performance gain are shown in the presentation

## References
* https://devblogs.nvidia.com/cublas-strided-batched-matrix-multiply/
* https://sahnimanas.github.io/post/anatomy-of-a-high-performance-convolution/
