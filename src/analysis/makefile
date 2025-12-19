PYTHON := $(shell command -v python3.11 || command -v python)

area_plot:
	$(PYTHON) produce_area_plot.py --device 3050ti --area 1 2 3 6     --pairs 1
	$(PYTHON) produce_area_plot.py --device 5090   --area 1 2 3 6     --pairs 1
	$(PYTHON) produce_area_plot.py --device A100   --area 1 2 3 4 5 6 --pairs 1
	$(PYTHON) produce_area_plot.py --device 3050ti --area 1 2 3       --pairs 512
	$(PYTHON) produce_area_plot.py --device 5090   --area 1 2 3       --pairs 512
	$(PYTHON) produce_area_plot.py --device A100   --area 1 2 3       --pairs 512

area_slowdown:
	$(PYTHON) get_area_decrease.py --device 3050ti --area 1 2 3 6     > "data/area_decrease/3050ti_areas_[1, 2, 3, 6].txt"
	$(PYTHON) get_area_decrease.py --device 5090   --area 1 2 3 6     > "data/area_decrease/5090_areas_[1, 2, 3, 6].txt"
	$(PYTHON) get_area_decrease.py --device A100   --area 1 2 3 4 5 6 > "data/area_decrease/A100_areas_[1, 2, 3, 4, 5, 6].txt"

fabric_ratio:
	$(PYTHON) get_fabric_ratios.py --device 3050ti --area 1 2 3 --pairs 1 2 4 8 16 32 64 128 256 512 > "data/fabric_ratios/3050ti_areas_[1, 2, 3]_pairs_[1, 2, 4, 8, 16, 32, 64, 128, 256, 512].txt"
	$(PYTHON) get_fabric_ratios.py --device 5090   --area 1 2 3 --pairs 1 2 4 8 16 32 64 128 256 512 > "data/fabric_ratios/5090_areas_[1, 2, 3]_pairs_[1, 2, 4, 8, 16, 32, 64, 128, 256, 512].txt"
	$(PYTHON) get_fabric_ratios.py --device A100   --area 1 2 3 --pairs 1 2 4 8 16 32 64 128 256 512 > "data/fabric_ratios/A100_areas_[1, 2, 3]_pairs_[1, 2, 4, 8, 16, 32, 64, 128, 256, 512].txt"
	$(PYTHON) get_fabric_ratios.py --device A100   --area 4 5   --pairs 1                            > "data/fabric_ratios/A100_areas_[4, 5]_pairs_[1].txt"

overhead_ratio:
	$(PYTHON) get_overhead_ratios.py --device 3050ti --area 1 2 3 --pairs 1 2 4 8 16 32 64 128 256 512 > "data/overhead_ratios/3050ti_areas_[1, 2, 3]_pairs_[1, 2, 4, 8, 16, 32, 64, 128, 256, 512].txt"
	$(PYTHON) get_overhead_ratios.py --device 5090   --area 1 2 3 --pairs 1 2 4 8 16 32 64 128 256 512 > "data/overhead_ratios/5090_areas_[1, 2, 3]_pairs_[1, 2, 4, 8, 16, 32, 64, 128, 256, 512].txt"
	$(PYTHON) get_overhead_ratios.py --device A100   --area 1 2 3 --pairs 1 2 4 8 16 32 64 128 256 512 > "data/overhead_ratios/A100_areas_[1, 2, 3]_pairs_[1, 2, 4, 8, 16, 32, 64, 128, 256, 512].txt"
	$(PYTHON) get_overhead_ratios.py --device A100   --area 4 5   --pairs 1                            > "data/overhead_ratios/A100_areas_[4, 5]_pairs_[1].txt"

pair_increase:
	$(PYTHON) get_pair_increase.py --device 3050ti --area 1 2 3     > "data/pair_increase/3050ti_areas_[1, 2, 3].txt"
	$(PYTHON) get_pair_increase.py --device 5090   --area 1 2 3     > "data/pair_increase/5090_areas_[1, 2, 3].txt"
	$(PYTHON) get_pair_increase.py --device A100   --area 1 2 3 4 5 > "data/pair_increase/A100_areas_[1, 2, 3, 4, 5].txt"

sync_percent:
	$(PYTHON) check_sync.py --device 3050ti --area 1 2 3     > "data/sync_percent/3050ti_areas_[1, 2, 3].txt"
	$(PYTHON) check_sync.py --device 5090   --area 1 2 3     > "data/sync_percent/5090_areas_[1, 2, 3].txt"
	$(PYTHON) check_sync.py --device A100   --area 1 2 3 4 5 > "data/sync_percent/A100_areas_[1, 2, 3, 4, 5].txt"

.PHONY: area_plot area_slowdown fabric_ratio overhead_ratio pair_increase sync_percent
