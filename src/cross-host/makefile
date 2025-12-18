NVCC = nvcc
NVCCFLAGS = -std=c++17 --expt-relaxed-constexpr --extended-lambda --compiler-options '-std=c++20'
INCLUDES = -I./ -I../nlohmann

SRC_CDA = ../warmup_kernel.cu cda_launcher.cu
SRC_GDR = ../warmup_kernel.cu gdr_launcher.cu
TARGET_CDA = bin_cda_main
TARGET_GDR = bin_gdr_main
PYTHON := $(shell command -v python3 || command -v python)

all: $(TARGET_CDA) $(TARGET_GDR)

$(TARGET_CDA): 
	$(NVCC) $(CFLAGS) $(INCLUDES) -o $(TARGET_CDA) $(SRC_CDA)

$(TARGET_GDR):
	@$(NVCC) $(CFLAGS) $(INCLUDES) -o $(TARGET_GDR) $(SRC_GDR) -lgdrapi -L/usr/lib64 || (echo "GDR API library not found. Please install the GDR API development package." && exit 1)

clean:
	rm -f $(TARGET_CDA) $(TARGET_GDR) *.o

benchmark-5090: $(TARGET_CDA)
	$(PYTHON) benchmark.py --max-host-mem=20000 --max-gpu-mem=30000 --fix-num-pairs=1
	
benchmark-A100: $(TARGET_CDA)
	$(PYTHON) cda_benchmark.py --max-host-mem=4000 --max-gpu-mem=2000 --fix-num-pairs=1
	${PYTHON} gdr_benchmark.py --max-host-mem=4000 --max-gpu-mem=2000 --fix-num-pairs=1

benchmark-3050ti: $(TARGET_CDA)
	$(PYTHON) benchmark.py --max-host-mem=4000 --max-gpu-mem=2000 --fix-num-pairs=1

.PHONY: all clean benchmark-5090 benchmark-A100 benchmark-3050ti
