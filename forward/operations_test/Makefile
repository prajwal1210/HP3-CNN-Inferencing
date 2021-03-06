
PROJECT_DIR = $(shell dirname $(shell dirname $(PWD)))
NVCC = nvcc
CXXFLAGS = -std=c++14
CUBLAS = /usr/local/cuda-10.1/man/man7/libcublas.so.7 #Have to make sure the libcublas.so.7 file is here, otherwise provide the proper path
OPENCV_LIB_FLAGS = $(shell pkg-config --cflags --libs opencv)
CUDA_LIB_FLAGS = -lcudnn -lcublas -lcufft -arch=sm_35 -rdc=true -O3
OPERATIONS_LIBRARY = $(shell find $(PROJECT_DIR) -name operations.cc)
DIRECT_CONV = $(shell find $(PROJECT_DIR) -name direct_conv.cu)
FFT_FILE = fft_fast.cu
FFT_KERNEL = $(shell find $(PROJECT_DIR) -name $(FFT_FILE))
WINOGRAD_FILE = winograd_fast.cu
WINOGRAD_KERNEL = $(shell find $(PROJECT_DIR) -name $(WINOGRAD_FILE))
IM2COL_FILE = im2col.cu
IM2COL_KERNEL = $(shell find $(PROJECT_DIR) -name $(IM2COL_FILE))

.PHONY : all clean

all : test

test: operations_test.cpp $(OPERATIONS_LIBRARY) $(FFT_KERNEL) $(DIRECT_CONV) $(WINOGRAD_KERNEL) $(IM2COL_KERNEL)
	$(NVCC) -I $(PROJECT_DIR) -L $(CUBLAS) $(CXXFLAGS) $? $(OPENCV_LIB_FLAGS) $(CUDA_LIB_FLAGS) -o $@

clean:
	rm -f test
	rm -f *.png
	rm -f *.txt

run_cudnn:
	./test
	python operations_test.py

run_direct:
	./test "DIRECT"
	python operations_test.py 

run_fft:
	./test "FFT"
	python operations_test.py

run_winograd:
	./test "WINOGRAD"
	python operations_test.py

run_im2col:
	./test "IM2COL"
	python operations_test.py
