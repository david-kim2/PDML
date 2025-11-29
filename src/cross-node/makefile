NVCC = nvcc
NVSHMEM_HOME ?= $(shell echo $$NVSHMEM_HOME)

ifeq ($(NVSHMEM_HOME),)
    $(error NVSHMEM_HOME is not set. Please load the nvshmem module or set NVSHMEM_HOME)
endif

NVCCFLAGS = -std=c++17 --expt-relaxed-constexpr --extended-lambda -arch=sm_80 -rdc=true
INCLUDES = -I./ -I../nlohmann -I$(NVSHMEM_HOME)/include
LDFLAGS = -L$(NVSHMEM_HOME)/lib -lnvshmem_host -lnvshmem_device -lcudart

SRC = launcher.cu echo_kernel.cu
TARGET = bin_main

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $(TARGET) $(SRC) $(LDFLAGS)

clean:
	rm -f $(TARGET) *.o
	rm -rf data/

test: $(TARGET)
	@echo "Testing with 1KB message, 1 pair, 5 runs"
	srun -N 2 -n 2 --gpus-per-task=1 ./bin_main 1024 1 5

.PHONY: all clean test