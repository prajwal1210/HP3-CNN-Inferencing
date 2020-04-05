NVCC=nvcc
CFLAGS=-std=c++11

sample: header.h kernel.cu host.cu
	$(NVCC) $(CFLAGS) kernel.cu host.cu -o sample

run: sample
	echo "29" | ./sample

clean:
	rm -f sample
