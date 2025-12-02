NVCC = nvcc
NVCCFLAGS = -std=c++17 --expt-relaxed-constexpr --extended-lambda --compiler-options '-std=c++20'
INCLUDES = -I./ -I../nlohmann

SRC = ../warmup_kernel.cu launcher.cu echo_kernel.cu
TARGET = bin_main

all: $(TARGET)

$(TARGET):
	$(NVCC) $(CFLAGS) $(INCLUDES) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET) *.o

benchmark-5090: $(TARGET)
	python benchmark.py --max-mem=64 --fix-num-pairs=1
	python benchmark.py --max-mem=64 --fix-num-pairs=2
	python benchmark.py --max-mem=64 --fix-num-pairs=4
	python benchmark.py --max-mem=64 --fix-num-pairs=8
	python benchmark.py --max-mem=64 --fix-num-pairs=16

benchmark-5090-further: $(TARGET)
	python benchmark.py --min-mem=65 --max-mem=8192 --fix-num-pairs=1
	python benchmark.py --min-mem=65 --max-mem=8192 --fix-num-pairs=2
	python benchmark.py --min-mem=65 --max-mem=8192 --fix-num-pairs=4
	python benchmark.py --min-mem=65 --max-mem=8192 --fix-num-pairs=8
	python benchmark.py --min-mem=65 --max-mem=8192 --fix-num-pairs=16

benchmark-5090-even-further: $(TARGET)
	python benchmark.py --min-mem=8193 --max-mem=1048576 --fix-num-pairs=1
	python benchmark.py --min-mem=8193 --max-mem=1048576 --fix-num-pairs=2
	python benchmark.py --min-mem=8193 --max-mem=1048576 --fix-num-pairs=4
	python benchmark.py --min-mem=8193 --max-mem=1048576 --fix-num-pairs=8
	python benchmark.py --min-mem=8193 --max-mem=1048576 --fix-num-pairs=16

benchmark-A100: $(TARGET)
	python benchmark.py --max-mem=64 --fix-num-pairs=1
	python benchmark.py --max-mem=64 --fix-num-pairs=2
	python benchmark.py --max-mem=64 --fix-num-pairs=4
	python benchmark.py --max-mem=64 --fix-num-pairs=8
	python benchmark.py --max-mem=64 --fix-num-pairs=16

benchmark-A100-further: $(TARGET)
	python benchmark.py --min-mem=65 --max-mem=8192 --fix-num-pairs=1
	python benchmark.py --min-mem=65 --max-mem=8192 --fix-num-pairs=2
	python benchmark.py --min-mem=65 --max-mem=8192 --fix-num-pairs=4
	python benchmark.py --min-mem=65 --max-mem=8192 --fix-num-pairs=8
	python benchmark.py --min-mem=65 --max-mem=8192 --fix-num-pairs=16

benchmark-A100-even-further: $(TARGET)
	python benchmark.py --min-mem=8193 --max-mem=1048576 --fix-num-pairs=1
	python benchmark.py --min-mem=8193 --max-mem=1048576 --fix-num-pairs=2
	python benchmark.py --min-mem=8193 --max-mem=1048576 --fix-num-pairs=4
	python benchmark.py --min-mem=8193 --max-mem=1048576 --fix-num-pairs=8
	python benchmark.py --min-mem=8193 --max-mem=1048576 --fix-num-pairs=16

benchmark-3050ti: $(TARGET)
	python benchmark.py --max-mem=64 --fix-num-pairs=1
	python benchmark.py --max-mem=64 --fix-num-pairs=2
	python benchmark.py --max-mem=64 --fix-num-pairs=4
	python benchmark.py --max-mem=64 --fix-num-pairs=8
	python benchmark.py --max-mem=64 --fix-num-pairs=16

benchmark-3050ti-further: $(TARGET)
	python benchmark.py --min-mem=65 --max-mem=8192 --fix-num-pairs=1
	python benchmark.py --min-mem=65 --max-mem=8192 --fix-num-pairs=2
	python benchmark.py --min-mem=65 --max-mem=8192 --fix-num-pairs=4
	python benchmark.py --min-mem=65 --max-mem=8192 --fix-num-pairs=8
	python benchmark.py --min-mem=65 --max-mem=8192 --fix-num-pairs=16

benchmark-3050ti-even-further: $(TARGET)
	python benchmark.py --min-mem=8193 --max-mem=1048576 --fix-num-pairs=1
	python benchmark.py --min-mem=8193 --max-mem=1048576 --fix-num-pairs=2
	python benchmark.py --min-mem=8193 --max-mem=1048576 --fix-num-pairs=4
	python benchmark.py --min-mem=8193 --max-mem=1048576 --fix-num-pairs=8
	python benchmark.py --min-mem=8193 --max-mem=1048576 --fix-num-pairs=16

.PHONY: all clean benchmark-5090 benchmark-A100 benchmark-3050ti benchmark-5090-further benchmark-A100-further benchmark-3050ti-further benchmark-5090-even-further benchmark-A100-even-further benchmark-3050ti-even-further
