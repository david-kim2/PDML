NVCC = nvcc
NVCCFLAGS = -std=c++17 --expt-relaxed-constexpr --extended-lambda --compiler-options '-std=c++20'
INCLUDES = -I./ -I../nlohmann

SRC = ../warmup_kernel.cu launcher.cu echo_kernel.cu
TARGET = bin_main
PYTHON := $(shell command -v python3.11 || command -v python3 || command -v python)

all: $(TARGET)

$(TARGET):
	$(NVCC) $(CFLAGS) $(INCLUDES) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET) *.o

benchmark-5090: $(TARGET)
	# $(PYTHON) benchmark.py --max-mem=1024 --fix-num-pairs=1
	# $(PYTHON) benchmark.py --max-mem=1024 --fix-num-pairs=2
	# $(PYTHON) benchmark.py --max-mem=1024 --fix-num-pairs=4
	# $(PYTHON) benchmark.py --max-mem=1024 --fix-num-pairs=8
	# $(PYTHON) benchmark.py --max-mem=1024 --fix-num-pairs=16
	# $(PYTHON) benchmark.py --max-mem=1024 --fix-num-pairs=32
	# $(PYTHON) benchmark.py --max-mem=1024 --fix-num-pairs=64
	# $(PYTHON) benchmark.py --max-mem=1024 --fix-num-pairs=128
	# $(PYTHON) benchmark.py --max-mem=1024 --fix-num-pairs=256
	# $(PYTHON) benchmark.py --max-mem=1024 --fix-num-pairs=512
	$(PYTHON) benchmark.py --max-mem=1024 --fix-num-pairs=1024
	
benchmark-A100: $(TARGET)
	$(PYTHON) benchmark.py --max-mem=4096 --fix-num-pairs=1
	$(PYTHON) benchmark.py --max-mem=4096 --fix-num-pairs=2
	$(PYTHON) benchmark.py --max-mem=4096 --fix-num-pairs=4
	$(PYTHON) benchmark.py --max-mem=4096 --fix-num-pairs=8
	$(PYTHON) benchmark.py --max-mem=4096 --fix-num-pairs=16
	$(PYTHON) benchmark.py --max-mem=4096 --fix-num-pairs=32
	$(PYTHON) benchmark.py --max-mem=4096 --fix-num-pairs=64
	$(PYTHON) benchmark.py --max-mem=4096 --fix-num-pairs=128
	$(PYTHON) benchmark.py --max-mem=4096 --fix-num-pairs=256
	$(PYTHON) benchmark.py --max-mem=4096 --fix-num-pairs=512

benchmark-3050ti: $(TARGET)
	# $(PYTHON) benchmark.py --max-mem=4096 --fix-num-pairs=1
	# $(PYTHON) benchmark.py --max-mem=4096 --fix-num-pairs=2
	# $(PYTHON) benchmark.py --max-mem=4096 --fix-num-pairs=4
	# $(PYTHON) benchmark.py --max-mem=4096 --fix-num-pairs=8
	# $(PYTHON) benchmark.py --max-mem=4096 --fix-num-pairs=16
	# $(PYTHON) benchmark.py --max-mem=4096 --fix-num-pairs=32
	$(PYTHON) benchmark.py --max-mem=4096 --fix-num-pairs=64
	$(PYTHON) benchmark.py --max-mem=4096 --fix-num-pairs=128
	$(PYTHON) benchmark.py --max-mem=4096 --fix-num-pairs=256
	$(PYTHON) benchmark.py --max-mem=4096 --fix-num-pairs=512

.PHONY: all clean benchmark-5090 benchmark-A100 benchmark-3050ti
