fabric_ratio:
	python3.11 get_fabric_ratios.py --device 3050ti --pairs 1 2 4 8 16 > "data/fabric_ratios/3050ti_areas_[1, 2, 3]_pairs_[1, 2, 4, 8, 16].txt"
	python3.11 get_fabric_ratios.py --device 5090 --pairs 1 2 4 8 16 > "data/fabric_ratios/5090_areas_[1, 2, 3]_pairs_[1, 2, 4, 8, 16].txt"
	python3.11 get_fabric_ratios.py --device A100 --pairs 1 2 4 8 16 > "data/fabric_ratios/A100_areas_[1, 2, 3]_pairs_[1, 2, 4, 8, 16].txt"

overhead_ratio:
	python3.11 get_overhead_ratios.py --device 3050ti --pairs 1 2 4 8 16 > "data/overhead_ratios/3050ti_areas_[1, 2, 3]_pairs_[1, 2, 4, 8, 16].txt"
	python3.11 get_overhead_ratios.py --device 5090 --pairs 1 2 4 8 16 > "data/overhead_ratios/5090_areas_[1, 2, 3]_pairs_[1, 2, 4, 8, 16].txt"
	python3.11 get_overhead_ratios.py --device A100 --pairs 1 2 4 8 16 > "data/overhead_ratios/A100_areas_[1, 2, 3]_pairs_[1, 2, 4, 8, 16].txt"

pair_increase:
	python3.11 get_pair_increase.py --device 3050ti > "data/pair_increase/3050ti_areas_[1, 2, 3].txt"
	python3.11 get_pair_increase.py --device 5090 > "data/pair_increase/5090_areas_[1, 2, 3].txt"
	python3.11 get_pair_increase.py --device A100 > "data/pair_increase/A100_areas_[1, 2, 3].txt"

area_slowdown:
	python3.11 get_area_decrease.py --device 3050ti > "data/area_decrease/3050ti_areas_[1, 2, 3].txt"
	python3.11 get_area_decrease.py --device 5090 > "data/area_decrease/5090_areas_[1, 2, 3].txt"
	python3.11 get_area_decrease.py --device A100 > "data/area_decrease/A100_areas_[1, 2, 3].txt"

area_plot:
	python3.11 produce_area_plot.py --device 3050ti --pairs 1
	python3.11 produce_area_plot.py --device 5090 --pairs 1
	python3.11 produce_area_plot.py --device A100 --pairs 1
	python3.11 produce_area_plot.py --device 3050ti --pairs 16
	python3.11 produce_area_plot.py --device 5090 --pairs 16
	python3.11 produce_area_plot.py --device A100 --pairs 16

.PHONY: fabric_ratio overhead_ratio pair_increase area_plot
