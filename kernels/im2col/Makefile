NVCC = nvcc
CXXFLAGS = -std=c++14
CUDA_LIB_FLAGS = -lcudnn -lcublas -lcufft -arch=sm_35 -rdc=true

test: im2col.cu im2coltest.cu
	$(NVCC) $(CXXFLAGS) $^ $(CUDA_LIB_FLAGS) -o $@
	./test

clean:
	rm test
	
