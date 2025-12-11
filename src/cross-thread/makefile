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
	# python benchmark.py --max-mem=1048576 --fix-num-pairs=1
	# python benchmark.py --max-mem=1048576 --fix-num-pairs=2
	# python benchmark.py --max-mem=1048576 --fix-num-pairs=4
	# python benchmark.py --max-mem=1048576 --fix-num-pairs=8
	# python benchmark.py --max-mem=1048576 --fix-num-pairs=16
	python benchmark.py --max-mem=1048576 --fix-num-pairs=32
	python benchmark.py --max-mem=1048576 --fix-num-pairs=64
	python benchmark.py --max-mem=1048576 --fix-num-pairs=128
	python benchmark.py --max-mem=1048576 --fix-num-pairs=256
	python benchmark.py --max-mem=1048576 --fix-num-pairs=512

benchmark-A100: $(TARGET)
	python benchmark.py --max-mem=1048576 --fix-num-pairs=1
	python benchmark.py --max-mem=1048576 --fix-num-pairs=2
	python benchmark.py --max-mem=1048576 --fix-num-pairs=4
	python benchmark.py --max-mem=1048576 --fix-num-pairs=8
	python benchmark.py --max-mem=1048576 --fix-num-pairs=16
	python benchmark.py --max-mem=1048576 --fix-num-pairs=32
	python benchmark.py --max-mem=1048576 --fix-num-pairs=64
	python benchmark.py --max-mem=1048576 --fix-num-pairs=128
	python benchmark.py --max-mem=1048576 --fix-num-pairs=256
	python benchmark.py --max-mem=1048576 --fix-num-pairs=512

benchmark-3050ti: $(TARGET)
	# python benchmark.py --max-mem=1048576 --fix-num-pairs=1
	# python benchmark.py --max-mem=1048576 --fix-num-pairs=2
	# python benchmark.py --max-mem=1048576 --fix-num-pairs=4
	# python benchmark.py --max-mem=1048576 --fix-num-pairs=8
	# python benchmark.py --max-mem=1048576 --fix-num-pairs=16
	python benchmark.py --max-mem=1048576 --fix-num-pairs=32
	python benchmark.py --max-mem=1048576 --fix-num-pairs=64
	python benchmark.py --max-mem=1048576 --fix-num-pairs=128
	python benchmark.py --max-mem=1048576 --fix-num-pairs=256
	python benchmark.py --max-mem=1048576 --fix-num-pairs=512

.PHONY: all clean benchmark-5090 benchmark-A100 benchmark-3050ti
