# PDML

Our project profiles data movement across the following hardware areas:

- `src/cross-thread`: Threads communicating with each other within a warp.
- `src/cross-warp`: Threads communicating across warps within a block.
- `src/cross-block`: Threads communicating across blocks within a GPU.
- `src/cross-gpu`: Threads communicating across GPUs within a node.
- `src/cross-gpu`: Threads communicating across GPUs across nodes.
- `src/cross-host`: Threads communicating between GPU and host CPU.

For each of these six areas, here are the corresponding commands to know:

- `make [all]` compiles the CUDA kernels into a binary file called `./bin_main`.
- `make clean` clears all binary files from a folder.
- `make benchmark-[device]` runs the benchmarking scripts for that specific device, note that the limits of these benchmarking scripts are dependent on relevant device features that can be listed by doing a run of `./bin_main`. This creates a folder of JSON outputs holding the 8 relevant timestamps in a subfolder called `data`.
- `cd plots && python produce_device_plot` processes the raw data from the benchmarking, and compiles them into a summary file also contained with the `data` folder. Additionally, this creates a per device plot with round trip latencies, round trip throughput, single trip latencies, overhead latencies, and fabic latencies that is also stored in the `plots` folder.
- `cd plots && python produce_comparison_plot` will create a cumulative plot comparing multiple devices in the same section if they exist, and store the resulting 5 plots (same as listed abve) in the `plots` folder as well.

Beyond these main 6 areas, there are two more relevant folders of code we wrote: `src/quantization` and `src/analysis` (`src/nlohmann` is just a C++ library we copied to be able to write to JSON, it can be found here: <https://github.com/nlohmann/json>).

- `src/quantization` is an extention to the `cross-block` area that has 4 different kernels (uint8, uint16, uint32, uint64) that test these varied quantizations across the same memory transfer metrics to see if there is a difference.
- `src/analysis` has a bunch of a post processing analysis scripts that are described below:

  - `make [all]` runs all of the following analyses at once.
  - `make area_plot` creates the 5 main graphs for a device across multiple hardware areas and stores them in `src/analysis/plots`
  - `make area_slowdown` calculates the average slowdown between different areas for all devices and stores them in `src/analysis/data/area_slowdown`.
  - `make fabric_latency` extracts the absolute fabric latency and relative ratio of fabric latency to single trip latency and stores them in `src/analysis/data/fabric_latency`.
  - `make min_lat_max_thr` extracts the minimum latency and max throughputs for each of the 14 hardware area + device combinations, and outputs it as a LaTeX table in `src/analysis/data/min_latency_max_throughput.txt`
  - `overhead_latency` extracts the absolute overhead latency and relative ratio of overhead latency to single trip latency and stores them in `src/analysis/data/overhead_latency`.
  - `make pair_speedup` calculates the average speedup between different # of pairs and stores them in `src/analysis/data/pair_speedup`.
  - `quantization_speedup` calcualtes the average spedeup between different quantization levels and stores them in `src/analysis/data/quantization_speedup`.
  - `sync_percent` compares the `sv_rx_ed` and `sv_tx_st` statistics for each run of every devices and hardware area to confirm that all servers finish receiving messages before any of them send messages back.

With these processing scripts, and all the raw data stored in this GitHub, you should be able to characterize our benchmarking experiments and expand on them pretty well.
