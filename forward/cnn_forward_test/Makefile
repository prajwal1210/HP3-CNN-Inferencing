
PROJECT_DIR = $(shell dirname $(shell dirname $(PWD)))
NVCC = nvcc
CXXFLAGS = -std=c++14
CUBLAS = /usr/local/cuda-10.1/man/man7/libcublas.so.7 #Have to make sure the libcublas.so.7 file is here, otherwise provide the proper path
OPENCV_LIB_FLAGS = $(shell pkg-config --cflags --libs opencv)
PROTOBUF_LIB_FLAGS = $(shell pkg-config --cflags --libs protobuf) 
CUDA_LIB_FLAGS = -lcudnn -lcublas -lcufft
OPERATIONS_LIBRARY = $(shell find $(PROJECT_DIR) -name operations.cc)
CNN_FORWARD_LIBRARY = $(shell find $(PROJECT_DIR) -name cnn_forward.cc)
DATA_UTIL_LIBRARY = $(shell find $(PROJECT_DIR) -name data_util.cc)
TRANSLATOR_LIBRARY = $(shell find $(PROJECT_DIR) -name translator.cc)
PROTO_LIBRARY = $(shell find $(PROJECT_DIR) -name network.pb.cc)
FFT_KERNEL = $(shell find $(PROJECT_DIR) -name fft.cu)

test: cnn_forward_test.cc $(OPERATIONS_LIBRARY) $(CNN_FORWARD_LIBRARY) $(DATA_UTIL_LIBRARY) $(TRANSLATOR_LIBRARY) $(PROTO_LIBRARY) $(FFT_KERNEL)
	$(NVCC) -I $(PROJECT_DIR) -L $(CUBLAS) $(CXXFLAGS) $? $(OPENCV_LIB_FLAGS) $(PROTOBUF_LIB_FLAGS) $(CUDA_LIB_FLAGS) -o $@

clean:
	rm test
	rm *.txt

run_cudnn:
	./test "$(PROJECT_DIR)/pretrained-models/vgg19.pb" 1>/dev/null
	python cnn_forward_test.py vgg
	@echo ""
	./test "$(PROJECT_DIR)/pretrained-models/alexnet.pb" 1>/dev/null
	python cnn_forward_test.py alex 
	
run_fft:
	./test "$(PROJECT_DIR)/pretrained-models/vgg19.pb" "FFT" 1>/dev/null
	python cnn_forward_test.py vgg
	@echo ""
	./test "$(PROJECT_DIR)/pretrained-models/alexnet.pb" "FFT" 1>/dev/null
	python cnn_forward_test.py alex 
	