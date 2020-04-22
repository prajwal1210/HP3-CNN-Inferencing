

## Direct convolution
#### Algorithm:
The direct convolution algorithm in its simplest form is expressed as 7 nested loops iterating along batch size **N**,  output channels **M**, output height **H_out**, output width **W_out**, and accumulating partial results across input channels **C**, kernel height **K**, kernel width **K**.
#### Input pre-processing:
* Input is of the form `batch_size * input_channels * Height * Width`
* **Pad the input:** Each Input layer is padded by the required number of zeroes on four sides.

#### Parallel Implementation:
* The parallelism in the implementation has four levels. As a summation is done over the iterating variables of 3 innermost loops, no further loop reductions are possible.
* Each thread will compute one element of one output feature map. 
* 2D thread blocks are used with each thread block computing a tile of **Tile_Width x Tile_Height** elements in one output feature map.
*  Blocks will be organized into a 3D grid,

        dim3 block( tile_width , tile_height , 1)
        dim3 grid( N , M , Z)
        
   * The last dimension Z  =  ( W_out / Tile_Width ) * ( H_out / Tile_Height ).   
* In the kernel ***direct_convolution***  , for each thread,
  * The batch number is calculated by `n = blockIdx.x`.
  * The output channel number is calculated by `m = blockIdx.y`.
  * The row number in the output array
  
         h = (blockIdx.z / W_grid)*tile_w + threadIdx.y
  * The column number in the output array     
  
          w = (blockIdx.z % W_grid)*tile_w + threadIdx.x
  * The result for the output element represented by the thread is obtained by normal convolution operation.

## Shortcomings
* Padding is to be done before copying the data onto GPU which adds to overhead time.
* Implementation assumes filters to be square.

## References
* https://www.elsevier.com/books/programming-massively-parallel-processors/kirk/978-0-12-811986-0
