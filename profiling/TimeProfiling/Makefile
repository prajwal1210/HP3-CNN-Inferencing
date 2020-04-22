
PROJECT_DIR = $(shell dirname $(shell dirname $(PWD)))
NVCC = nvcc
CXXFLAGS = -std=c++14
CUBLAS = /usr/local/cuda-10.1/man/man7/libcublas.so.7 #Have to make sure the libcublas.so.7 file is here, otherwise provide the proper path
OPENCV_LIB_FLAGS = $(shell pkg-config --cflags --libs opencv)
PROTOBUF_LIB_FLAGS = $(shell pkg-config --libs protobuf) 
CUDA_LIB_FLAGS = -lcudnn -lcublas -lcufft -arch=sm_35 -rdc=true -O3
OPERATIONS_LIBRARY = $(shell find $(PROJECT_DIR) -name operations.cc)
CNN_FORWARD_LIBRARY = $(shell find $(PROJECT_DIR) -name cnn_forward.cc)
DATA_UTIL_LIBRARY = $(shell find $(PROJECT_DIR) -name data_util.cc)
TRANSLATOR_LIBRARY = $(shell find $(PROJECT_DIR) -name translator.cc)
PROTO_LIBRARY = $(shell find $(PROJECT_DIR) -name network.pb.cc)
DIRECT_CONV = $(shell find $(PROJECT_DIR) -name direct_conv.cu)
FFT_FILE = fft_fast.cu
FFT_KERNEL = $(shell find $(PROJECT_DIR) -name $(FFT_FILE))
WINOGRAD_FILE = winograd_fast.cu
WINOGRAD_KERNEL = $(shell find $(PROJECT_DIR) -name $(WINOGRAD_FILE))
IM2COL_FILE = im2col.cu
IM2COL_KERNEL = $(shell find $(PROJECT_DIR) -name $(IM2COL_FILE))
NUM_RUNS = 3
TARGET = time_prof

.PHONY: all clean

all: $(TARGET)

$(TARGET): timeProfiler.cc $(OPERATIONS_LIBRARY) $(CNN_FORWARD_LIBRARY) $(DATA_UTIL_LIBRARY) $(TRANSLATOR_LIBRARY) $(PROTO_LIBRARY) $(FFT_KERNEL) $(DIRECT_CONV) $(WINOGRAD_KERNEL) $(IM2COL_KERNEL)
	$(NVCC) -I $(PROJECT_DIR) -L $(CUBLAS) $(CXXFLAGS) $? $(OPENCV_LIB_FLAGS) $(PROTOBUF_LIB_FLAGS) $(CUDA_LIB_FLAGS) -o $@

clean:
	rm -f $(TARGET)

run_vgg:
	rm -f logVGG.txt
	./$(TARGET) "VGG" "$(PROJECT_DIR)/pretrained-models/vgg19.pb" $(NUM_RUNS) "DIRECT" 1>/dev/null
	./$(TARGET) "VGG" "$(PROJECT_DIR)/pretrained-models/vgg19.pb" $(NUM_RUNS) "IM2COL" 1>/dev/null
	./$(TARGET) "VGG" "$(PROJECT_DIR)/pretrained-models/vgg19.pb" $(NUM_RUNS) "FFT" 1>/dev/null
	./$(TARGET) "VGG" "$(PROJECT_DIR)/pretrained-models/vgg19.pb" $(NUM_RUNS) "WINOGRAD" 1>/dev/null
	./$(TARGET) "VGG" "$(PROJECT_DIR)/pretrained-models/vgg19.pb" $(NUM_RUNS) 1>/dev/null

run_alex:
	rm -f logALEX.txt
	./$(TARGET) "ALEX" "$(PROJECT_DIR)/pretrained-models/alexnet.pb" $(NUM_RUNS) "DIRECT" 1>/dev/null
	./$(TARGET) "ALEX" "$(PROJECT_DIR)/pretrained-models/alexnet.pb" $(NUM_RUNS) "IM2COL" 1>/dev/null
	./$(TARGET) "ALEX" "$(PROJECT_DIR)/pretrained-models/alexnet.pb" $(NUM_RUNS) "FFT" 1>/dev/null
	./$(TARGET) "ALEX" "$(PROJECT_DIR)/pretrained-models/alexnet.pb" $(NUM_RUNS) 1>/dev/null