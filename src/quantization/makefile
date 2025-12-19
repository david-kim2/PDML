NVCC = nvcc
NVCCFLAGS = -std=c++17 --expt-relaxed-constexpr --extended-lambda --compiler-options '-std=c++20'
INCLUDES = -I./ -I../nlohmann

SRC_UINT8 = ../warmup_kernel.cu uint8_launcher.cu uint8_kernel.cu
SRC_UINT16 = ../warmup_kernel.cu uint16_launcher.cu uint16_kernel.cu
SRC_UINT32 = ../warmup_kernel.cu uint32_launcher.cu uint32_kernel.cu
SRC_UINT64 = ../warmup_kernel.cu uint64_launcher.cu uint64_kernel.cu

TARGET_UINT8 = bin_main_uint8
TARGET_UINT16 = bin_main_uint16
TARGET_UINT32 = bin_main_uint32
TARGET_UINT64 = bin_main_uint64

PYTHON := $(shell command -v python3 || command -v python)

all: $(TARGET_UINT8) $(TARGET_UINT16) $(TARGET_UINT32) $(TARGET_UINT64)

$(TARGET_UINT8):
	$(NVCC) $(CFLAGS) $(INCLUDES) -o $(TARGET_UINT8) $(SRC_UINT8)

$(TARGET_UINT16):
	$(NVCC) $(CFLAGS) $(INCLUDES) -o $(TARGET_UINT16) $(SRC_UINT16)

$(TARGET_UINT32):
	$(NVCC) $(CFLAGS) $(INCLUDES) -o $(TARGET_UINT32) $(SRC_UINT32)

$(TARGET_UINT64):
	$(NVCC) $(CFLAGS) $(INCLUDES) -o $(TARGET_UINT64) $(SRC_UINT64)

clean:
	rm -f $(TARGET_UINT8) $(TARGET_UINT16) $(TARGET_UINT32) $(TARGET_UINT64) *.o

benchmark-5090: $(TARGET_UINT8) $(TARGET_UINT16) $(TARGET_UINT32) $(TARGET_UINT64)
	$(PYTHON) uint8_benchmark.py  --max-mem=4096 --fix-num-pairs=1024
	$(PYTHON) uint16_benchmark.py --max-mem=4096 --fix-num-pairs=1024
	$(PYTHON) uint32_benchmark.py --max-mem=4096 --fix-num-pairs=1024
	$(PYTHON) uint64_benchmark.py --max-mem=4096 --fix-num-pairs=1024

benchmark-A100: $(TARGET_UINT8) $(TARGET_UINT16) $(TARGET_UINT32) $(TARGET_UINT64)
	$(PYTHON) uint8_benchmark.py  --max-mem=4096 --fix-num-pairs=1024
	$(PYTHON) uint16_benchmark.py --max-mem=4096 --fix-num-pairs=1024
	$(PYTHON) uint32_benchmark.py --max-mem=4096 --fix-num-pairs=1024
	$(PYTHON) uint64_benchmark.py --max-mem=4096 --fix-num-pairs=1024

benchmark-3050ti: $(TARGET_UINT8) $(TARGET_UINT16) $(TARGET_UINT32) $(TARGET_UINT64)
	$(PYTHON) uint8_benchmark.py  --max-mem=4096 --fix-num-pairs=1024
	$(PYTHON) uint16_benchmark.py --max-mem=4096 --fix-num-pairs=1024
	$(PYTHON) uint32_benchmark.py --max-mem=4096 --fix-num-pairs=1024
	$(PYTHON) uint64_benchmark.py --max-mem=4096 --fix-num-pairs=1024

.PHONY: all clean benchmark-5090 benchmark-A100 benchmark-3050ti
