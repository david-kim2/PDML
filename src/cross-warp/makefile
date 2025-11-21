NVCC = nvcc
NVCCFLAGS = -std=c++17 --expt-relaxed-constexpr --extended-lambda \
            --compiler-options '-std=c++20'

INCLUDES = -I./

SRC = warmup_kernel.cu launcher.cu echo_kernel.cu
TARGET = main

all: $(TARGET)

$(TARGET):
	$(NVCC) $(CFLAGS) $(INCLUDES) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET) *.o

.PHONY: all clean
