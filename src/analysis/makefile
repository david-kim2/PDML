PYTHON := $(shell command -v python3.11 || command -v python)

all: area_slowdown fabric_latency min_lat_max_thr overhead_latency pair_speedup quantization_speedup sync_percent

area_plot:
	$(PYTHON) produce_area_plot.py --device 3050ti --area 1 2 3 6     --pairs 1
	$(PYTHON) produce_area_plot.py --device 5090   --area 1 2 3 6     --pairs 1
	$(PYTHON) produce_area_plot.py --device A100   --area 1 2 3 4 5 6 --pairs 1
	$(PYTHON) produce_area_plot.py --device 3050ti --area 1 2 3       --pairs 512
	$(PYTHON) produce_area_plot.py --device 5090   --area 1 2 3       --pairs 512
	$(PYTHON) produce_area_plot.py --device A100   --area 1 2 3       --pairs 512

area_slowdown:
	$(PYTHON) get_area_slowdown.py --device 3050ti --area 1 2 3 6     > "data/area_slowdown/3050ti_areas_[1, 2, 3, 6].txt"
	$(PYTHON) get_area_slowdown.py --device 5090   --area 1 2 3 6     > "data/area_slowdown/5090_areas_[1, 2, 3, 6].txt"
	$(PYTHON) get_area_slowdown.py --device A100   --area 1 2 3 4 5 6 > "data/area_slowdown/A100_areas_[1, 2, 3, 4, 5, 6].txt"

fabric_latency:
	$(PYTHON) get_fabric_latency.py --device 3050ti --area 1 2 3     --pairs 1 2 4 8 16 32 64 128 256 512 1024 \
		> "data/fabric_latency/3050ti_areas_[1, 2, 3]_pairs_[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024].txt"
	$(PYTHON) get_fabric_latency.py --device 5090   --area 1 2 3     --pairs 1 2 4 8 16 32 64 128 256 512 1024 \
		> "data/fabric_latency/5090_areas_[1, 2, 3]_pairs_[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024].txt"
	$(PYTHON) get_fabric_latency.py --device A100   --area 1 2 3 4 5 --pairs 1 2 4 8 16 32 64 128 256 512 1024 \
		> "data/fabric_latency/A100_areas_[1, 2, 3, 4, 5]_pairs_[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024].txt"

min_lat_max_thr:
	$(PYTHON) min_lat_max_thr.py > "data/min_latency_max_throughput.txt"

overhead_latency:
	$(PYTHON) get_overhead_latency.py --device 3050ti --area 1 2 3     --pairs 1 2 4 8 16 32 64 128 256 512 1024 \
		> "data/overhead_latency/3050ti_areas_[1, 2, 3]_pairs_[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024].txt"
	$(PYTHON) get_overhead_latency.py --device 5090   --area 1 2 3     --pairs 1 2 4 8 16 32 64 128 256 512 1024 \
		> "data/overhead_latency/5090_areas_[1, 2, 3]_pairs_[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024].txt"
	$(PYTHON) get_overhead_latency.py --device A100   --area 1 2 3 4 5 --pairs 1 2 4 8 16 32 64 128 256 512 1024 \
		> "data/overhead_latency/A100_areas_[1, 2, 3, 4, 5]_pairs_[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024].txt"

pair_speedup:
	$(PYTHON) get_pair_speedup.py --device 3050ti --area 1 2 3     > "data/pair_speedup/3050ti_areas_[1, 2, 3].txt"
	$(PYTHON) get_pair_speedup.py --device 5090   --area 1 2 3     > "data/pair_speedup/5090_areas_[1, 2, 3].txt"
	$(PYTHON) get_pair_speedup.py --device A100   --area 1 2 3 4 5 > "data/pair_speedup/A100_areas_[1, 2, 3, 4, 5].txt"

quantization_speedup:
	$(PYTHON) get_quantization_speedup.py --device 3050ti > "data/quantization_speedup/3050ti.txt"
	$(PYTHON) get_quantization_speedup.py --device 5090   > "data/quantization_speedup/5090.txt"
	$(PYTHON) get_quantization_speedup.py --device A100   > "data/quantization_speedup/A100.txt"

sync_percent:
	$(PYTHON) check_sync_percent.py --device 3050ti --area 1 2 3     > "data/sync_percent/3050ti_areas_[1, 2, 3].txt"
	$(PYTHON) check_sync_percent.py --device 5090   --area 1 2 3     > "data/sync_percent/5090_areas_[1, 2, 3].txt"
	$(PYTHON) check_sync_percent.py --device A100   --area 1 2 3 4 5 > "data/sync_percent/A100_areas_[1, 2, 3, 4, 5].txt"

.PHONY: area_plot area_slowdown fabric_latency min_lat_max_thr overhead_latency pair_speedup quantization_speedup sync_percent
