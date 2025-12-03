NVCC = nvcc
NVSHMEM_HOME = $(HOME)/libnvshmem-linux-x86_64-3.4.5_cuda12-archive

NVCCFLAGS = -std=c++17 --expt-relaxed-constexpr --extended-lambda -rdc=true -ccbin=g++ -arch=sm_80
INCLUDES = -I./ -I../nlohmann -I$(NVSHMEM_HOME)/include
LDFLAGS = -L$(NVSHMEM_HOME)/lib -lnvshmem_host $(NVSHMEM_HOME)/lib/libnvshmem_device.a -lcuda -Xlinker -rpath,$(NVSHMEM_HOME)/lib

SRC = ../warmup_kernel.cu launcher.cu echo_kernel.cu
TARGET = bin_main

all: $(TARGET)

$(TARGET):
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $(TARGET) $(SRC) $(LDFLAGS)

clean:
	rm -f $(TARGET) *.o
	rm -rf data/

benchmark: $(TARGET)
	python benchmark.py --max-mem=1024 --fix-num-pairs=1
	python benchmark.py --max-mem=1024 --fix-num-pairs=2
	python benchmark.py --max-mem=1024 --fix-num-pairs=4
	python benchmark.py --max-mem=1024 --fix-num-pairs=8
	python benchmark.py --max-mem=1024 --fix-num-pairs=16
	python benchmark.py --max-mem=1024 --fix-num-pairs=32

.PHONY: all clean benchmark
