
PROJECT_DIR = $(shell dirname $(shell dirname $(PWD)))
NVCC = nvcc
CXXFLAGS = -std=c++14
CUBLAS = /usr/local/cuda-10.1/man/man7/libcublas.so.7 #Have to make sure the libcublas.so.7 file is here, otherwise provide the proper path
OPENCV_LIB_FLAGS = $(shell pkg-config --cflags --libs opencv)
CUDA_LIB_FLAGS = -lcudnn -lcublas
OPERATIONS_LIBRARY = $(shell find $(PROJECT_DIR) -name operations.cc)
DIRECT_CONV = $(shell find $(PROJECT_DIR) -name direct_conv.cu)

test: operations_test.cpp $(OPERATIONS_LIBRARY) $(DIRECT_CONV)
	$(NVCC) -I $(PROJECT_DIR) -L $(CUBLAS) $(CXXFLAGS) $? $(OPENCV_LIB_FLAGS) $(CUDA_LIB_FLAGS) -o $@

clean:
	rm test
	rm *.png
	rm *.txt

run:
	./test
	python operations_test.py
	