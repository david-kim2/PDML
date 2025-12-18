NVCC = nvcc
NVCCFLAGS = -std=c++17 --expt-relaxed-constexpr --extended-lambda --compiler-options '-std=c++20'
INCLUDES = -I./ -I../nlohmann

SRC = ../warmup_kernel.cu launcher.cu echo_kernel.cu
TARGET = bin_main
PYTHON := $(shell command -v python3 || command -v python)

all: $(TARGET)

$(TARGET):
	$(NVCC) $(CFLAGS) $(INCLUDES) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET) *.o

benchmark-5090: $(TARGET)
	$(PYTHON) benchmark.py                   --max-mem=1000000  --fix-num-pairs=1
	$(PYTHON) benchmark.py --min-mem=1000000 --max-mem=20000000 --fix-num-pairs=1   --num-runs=10
	$(PYTHON) benchmark.py                   --max-mem=1000000  --fix-num-pairs=2
	$(PYTHON) benchmark.py --min-mem=1000000 --max-mem=20000000 --fix-num-pairs=2   --num-runs=10
	$(PYTHON) benchmark.py                   --max-mem=1000000  --fix-num-pairs=4
	$(PYTHON) benchmark.py --min-mem=1000000 --max-mem=20000000 --fix-num-pairs=4   --num-runs=10
	$(PYTHON) benchmark.py                   --max-mem=1000000  --fix-num-pairs=8
	$(PYTHON) benchmark.py --min-mem=1000000 --max-mem=20000000 --fix-num-pairs=8   --num-runs=10
	$(PYTHON) benchmark.py                   --max-mem=1000000  --fix-num-pairs=16
	$(PYTHON) benchmark.py --min-mem=1000000 --max-mem=20000000 --fix-num-pairs=16  --num-runs=10
	$(PYTHON) benchmark.py                   --max-mem=1000000  --fix-num-pairs=32
	$(PYTHON) benchmark.py --min-mem=1000000 --max-mem=20000000 --fix-num-pairs=32  --num-runs=10
	$(PYTHON) benchmark.py                   --max-mem=1000000  --fix-num-pairs=64
	$(PYTHON) benchmark.py --min-mem=1000000 --max-mem=20000000 --fix-num-pairs=64  --num-runs=10
	$(PYTHON) benchmark.py                   --max-mem=1000000  --fix-num-pairs=128
	$(PYTHON) benchmark.py --min-mem=1000000 --max-mem=20000000 --fix-num-pairs=128 --num-runs=10
	$(PYTHON) benchmark.py                   --max-mem=1000000  --fix-num-pairs=256
	$(PYTHON) benchmark.py --min-mem=1000000 --max-mem=20000000 --fix-num-pairs=256 --num-runs=10
	$(PYTHON) benchmark.py                   --max-mem=1000000  --fix-num-pairs=512
	$(PYTHON) benchmark.py --min-mem=1000000 --max-mem=20000000 --fix-num-pairs=512 --num-runs=10

benchmark-A100: $(TARGET)
	# $(PYTHON) benchmark.py                   --max-mem=1000000  --fix-num-pairs=1
	# $(PYTHON) benchmark.py --min-mem=1000000 --max-mem=20000000 --fix-num-pairs=1   --num-runs=10
	# $(PYTHON) benchmark.py                   --max-mem=1000000  --fix-num-pairs=2
	# $(PYTHON) benchmark.py --min-mem=1000000 --max-mem=20000000 --fix-num-pairs=2   --num-runs=10
	# $(PYTHON) benchmark.py                   --max-mem=1000000  --fix-num-pairs=4
	# $(PYTHON) benchmark.py --min-mem=1000000 --max-mem=20000000 --fix-num-pairs=4   --num-runs=10
	# $(PYTHON) benchmark.py                   --max-mem=1000000  --fix-num-pairs=8
	# $(PYTHON) benchmark.py --min-mem=1000000 --max-mem=20000000 --fix-num-pairs=8   --num-runs=10
	# $(PYTHON) benchmark.py                   --max-mem=1000000  --fix-num-pairs=16
	# $(PYTHON) benchmark.py --min-mem=1000000 --max-mem=20000000 --fix-num-pairs=16  --num-runs=10
	# $(PYTHON) benchmark.py                   --max-mem=1000000  --fix-num-pairs=32
	# $(PYTHON) benchmark.py --min-mem=1000000 --max-mem=20000000 --fix-num-pairs=32  --num-runs=10
	$(PYTHON) benchmark.py                   --max-mem=1000000  --fix-num-pairs=64
	$(PYTHON) benchmark.py --min-mem=1000000 --max-mem=20000000 --fix-num-pairs=64  --num-runs=10
	$(PYTHON) benchmark.py                   --max-mem=1000000  --fix-num-pairs=128
	$(PYTHON) benchmark.py --min-mem=1000000 --max-mem=20000000 --fix-num-pairs=128 --num-runs=10
	$(PYTHON) benchmark.py                   --max-mem=1000000  --fix-num-pairs=256
	$(PYTHON) benchmark.py --min-mem=1000000 --max-mem=20000000 --fix-num-pairs=256 --num-runs=10
	$(PYTHON) benchmark.py                   --max-mem=1000000  --fix-num-pairs=512
	$(PYTHON) benchmark.py --min-mem=1000000 --max-mem=20000000 --fix-num-pairs=512 --num-runs=10

benchmark-3050ti: $(TARGET)
	$(PYTHON) benchmark.py                   --max-mem=1000000  --fix-num-pairs=1
	$(PYTHON) benchmark.py --min-mem=1000000 --max-mem=20000000 --fix-num-pairs=1   --num-runs=10
	$(PYTHON) benchmark.py                   --max-mem=1000000  --fix-num-pairs=2
	$(PYTHON) benchmark.py --min-mem=1000000 --max-mem=20000000 --fix-num-pairs=2   --num-runs=10
	$(PYTHON) benchmark.py                   --max-mem=1000000  --fix-num-pairs=4
	$(PYTHON) benchmark.py --min-mem=1000000 --max-mem=20000000 --fix-num-pairs=4   --num-runs=10
	$(PYTHON) benchmark.py                   --max-mem=1000000  --fix-num-pairs=8
	$(PYTHON) benchmark.py --min-mem=1000000 --max-mem=20000000 --fix-num-pairs=8   --num-runs=10
	$(PYTHON) benchmark.py                   --max-mem=1000000  --fix-num-pairs=16
	$(PYTHON) benchmark.py --min-mem=1000000 --max-mem=20000000 --fix-num-pairs=16  --num-runs=10
	$(PYTHON) benchmark.py                   --max-mem=1000000  --fix-num-pairs=32
	$(PYTHON) benchmark.py --min-mem=1000000 --max-mem=20000000 --fix-num-pairs=32  --num-runs=10
	$(PYTHON) benchmark.py                   --max-mem=1000000  --fix-num-pairs=64
	$(PYTHON) benchmark.py --min-mem=1000000 --max-mem=20000000 --fix-num-pairs=64  --num-runs=10
	$(PYTHON) benchmark.py                   --max-mem=1000000  --fix-num-pairs=128
	$(PYTHON) benchmark.py --min-mem=1000000 --max-mem=20000000 --fix-num-pairs=128 --num-runs=10
	$(PYTHON) benchmark.py                   --max-mem=1000000  --fix-num-pairs=256
	$(PYTHON) benchmark.py --min-mem=1000000 --max-mem=20000000 --fix-num-pairs=256 --num-runs=10
	$(PYTHON) benchmark.py                   --max-mem=1000000  --fix-num-pairs=512
	$(PYTHON) benchmark.py --min-mem=1000000 --max-mem=20000000 --fix-num-pairs=512 --num-runs=10

.PHONY: all clean benchmark-5090 benchmark-A100 benchmark-3050ti
