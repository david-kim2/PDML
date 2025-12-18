NVCC = nvcc
NVCCFLAGS = -std=c++17 --expt-relaxed-constexpr --extended-lambda --compiler-options '-std=c++20'
INCLUDES = -I./ -I../nlohmann

SRC = ../warmup_kernel.cu launcher.cu
TARGET = bin_main
PYTHON := $(shell command -v python3 || command -v python)

all: $(TARGET)

$(TARGET):
	$(NVCC) $(CFLAGS) $(INCLUDES) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET) *.o

benchmark-5090: $(TARGET)
	$(PYTHON) benchmark.py --max-host-mem=20000 --max-gpu-mem=30000 --fix-num-pairs=1
	
benchmark-A100: $(TARGET)
	$(PYTHON) benchmark.py --max-host-mem=4000 --max-gpu-mem=2000 --fix-num-pairs=1

benchmark-3050ti: $(TARGET)
	$(PYTHON) benchmark.py --max-host-mem=4000 --max-gpu-mem=2000 --fix-num-pairs=1

.PHONY: all clean benchmark-5090 benchmark-A100 benchmark-3050ti
