#!/bin/bash

BINARY="${1:mem_prof}"


echo "Running for Direct Convolution"
nvidia-smi --query-gpu=memory.used --format=csv -lms 10 > memory_direct.txt &
./$BINARY "../../pretrained-models/vgg19.pb" "DIRECT" 1>/dev/null
sleep 1
pkill -9 nvidia-smi

echo "Running for Im2Col Convolution"
nvidia-smi --query-gpu=memory.used --format=csv -lms 10 > memory_im2col.txt &
./$BINARY "../../pretrained-models/vgg19.pb" "IM2COL" 1>/dev/null
sleep 1
pkill -9 nvidia-smi

echo "Running for FFT Convolution"
nvidia-smi --query-gpu=memory.used --format=csv -lms 10 > memory_fft.txt &
./$BINARY "../../pretrained-models/vgg19.pb" "FFT" 1>/dev/null
sleep 1
pkill -9 nvidia-smi

echo "Running for Winograd Convolution"
nvidia-smi --query-gpu=memory.used --format=csv -lms 10 > memory_winograd.txt &
./$BINARY "../../pretrained-models/vgg19.pb" "WINOGRAD" 1>/dev/null
sleep 1
pkill -9 nvidia-smi

echo "Running for CUDNN Convolution"
nvidia-smi --query-gpu=memory.used --format=csv -lms 10 > memory_cudnn.txt &
./$BINARY "../../pretrained-models/vgg19.pb" 1>/dev/null
sleep 1
pkill -9 nvidia-smi
set +x
