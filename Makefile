SUBDIR_TESTING = forward/cnn_forward_test/ forward/batch_test forward/operations_test 
SUBDIR_PROFILING = profiling/TimeProfiling profiling/MemoryUsageProfiling
FFT_FILE = fft_fast.cu
WINOGRAD_FILE = winograd_fast.cu
MAKE = make
TARGETS = clean all

.PHONY : profilers tests $(SUBDIR_PROFILING) $(SUBDIR_TESTING)

profilers: $(SUBDIR_PROFILING)

$(SUBDIR_PROFILING):
	$(MAKE) -C $@ $(TARGETS) PROJECT_DIR=$(PWD) FFT_FILE=$(FFT_FILE) WINOGRAD_FILE=$(WINOGRAD_FILE)

tests: $(SUBDIR_TESTING)

$(SUBDIR_TESTING):
	$(MAKE) -C $@ $(TARGETS) PROJECT_DIR=$(PWD) FFT_FILE=$(FFT_FILE) WINOGRAD_FILE=$(WINOGRAD_FILE)