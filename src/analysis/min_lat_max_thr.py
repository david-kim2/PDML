import json
import os


def format_instruction(x, pos):
    if x >= 1e9:   return f"{x / 1e9:.2f} G"
    elif x >= 1e6: return f"{x / 1e6:.2f} M"
    elif x >= 1e3: return f"{x / 1e3:.2f} K"
    else:          return f"{int(x)}"


def format_bytes(x, pos):
    if x >= 1 << 30:   return f"{x / (1 << 30):.2f} GiB"
    elif x >= 1 << 20: return f"{x / (1 << 20):.2f} MiB"
    elif x >= 1 << 10: return f"{x / (1 << 10):.2f} KiB"
    else:              return f"{int(x)}B"


def time_format(x, pos):
    if x >= 1e9:   return f"{x / 1e9:.2f} s"
    elif x >= 1e6: return f"{x / 1e6:.2f} ms"
    elif x >= 1e3: return f"{x / 1e3:.2f} µs"
    else:          return f"{x:.2f} ns"


def min_latency(json_data):
    min_latency = float('inf')
    min_latency_entry = None
    for entry in json_data:
        latency = entry["metrics"]["round_trip_latency_avg"]
        if latency < min_latency:
            min_latency = latency
            min_latency_entry = entry
    return min_latency, min_latency_entry


def max_throughput(json_data):
    max_throughput = 0
    max_throughput_entry = None
    for entry in json_data:
        throughput = entry["metrics"]["round_trip_throughput_avg"]
        if throughput > max_throughput:
            max_throughput = throughput
            max_throughput_entry = entry
    return max_throughput, max_throughput_entry


def generate_latex_table_with_details(area_alias, reverse_hwd_alias):
    # Storage: results[hardware][area] = (latency, thr, min_entry, max_entry)
    results = {
        hwd: {area: {"lat": None, "lat_entry": None, "thr": None, "thr_entry": None} for area in area_alias.values()}
        for hwd in reverse_hwd_alias
    }

    # Collect data
    for area_id, area_name in area_alias.items():
        dir_name = f"../{area_name}/data"
        for hwd, hwd_dir in reverse_hwd_alias.items():
            json_name = f"{dir_name}/{hwd_dir}_metrics.json"
            if not os.path.exists(json_name):
                continue

            with open(json_name, "r") as f:
                json_data = json.load(f)

            min_lat, min_entry = min_latency(json_data)
            max_thr, max_entry = max_throughput(json_data)

            results[hwd][area_name]["lat"] = time_format(min_lat, None)
            results[hwd][area_name]["lat_entry"] = min_entry
            results[hwd][area_name]["thr"] = format_bytes(max_thr, None)
            results[hwd][area_name]["thr_entry"] = max_entry

    # ---- LaTeX output ----
    areas = list(area_alias.values())
    print(r"\renewcommand{\arraystretch}{2}")  # Make rows taller
    print(r"\begin{tabular}{|c|" + "c" * len(areas) + "|}")
    print(r"\hline")
    print("Device & " + " & ".join(a.replace("-", " ").title() for a in areas) + r" \\")
    print(r"\hline")

    # Throughput rows
    for hwd in reverse_hwd_alias:
        row = []
        for a in areas:
            thr = results[hwd][a]["thr"]
            entry = results[hwd][a]["thr_entry"]
            if thr and entry:
                # row.append(r"\shortstack[b]{" + f"{thr}\\\\{{\\scriptsize P={entry['num_pairs']}, @{format_bytes(entry['msg_size'], None)}}}" + "}")
                # swap
                row.append(r"\shortstack[b]{" + f"{thr}\\\\{{\\scriptsize @{format_bytes(entry['msg_size'], None)}, P={entry['num_pairs']}}}" + "}")
            else:
                row.append("--")
        print(f"{hwd} Throughput & " + " & ".join(row) + r" \\")
    print(r"\hline")

    # Latency rows
    for hwd in reverse_hwd_alias:
        row = []
        for a in areas:
            lat = results[hwd][a]["lat"]
            entry = results[hwd][a]["lat_entry"]
            if lat and entry:
                # row.append(r"\shortstack[b]{" + f"{lat}\\\\{{\\scriptsize P={entry['num_pairs']}, @{format_bytes(entry['msg_size'], None)}}}" + "}")
                # swap
                row.append(r"\shortstack[b]{" + f"{lat}\\\\{{\\scriptsize @{format_bytes(entry['msg_size'], None)}, P={entry['num_pairs']}}}" + "}")
            else:
                row.append("--")
        print(f"{hwd} Latency & " + " & ".join(row) + r" \\")
    print(r"\hline")
    print(r"\end{tabular}")

def generate_latex_table_with_details_transposed(area_alias, reverse_hwd_alias):
    # results[hardware][area]
    results = {
        hwd: {area: {"lat": None, "lat_entry": None, "thr": None, "thr_entry": None}
              for area in area_alias.values()}
        for hwd in reverse_hwd_alias
    }

    # Collect data
    for area_id, area_name in area_alias.items():
        dir_name = f"../{area_name}/data"
        for hwd, hwd_dir in reverse_hwd_alias.items():
            json_name = f"{dir_name}/{hwd_dir}_metrics.json"
            if not os.path.exists(json_name):
                continue

            with open(json_name, "r") as f:
                json_data = json.load(f)

            min_lat, min_entry = min_latency(json_data)
            max_thr, max_entry = max_throughput(json_data)

            results[hwd][area_name]["lat"] = time_format(min_lat, None)
            results[hwd][area_name]["lat_entry"] = min_entry
            results[hwd][area_name]["thr"] = format_bytes(max_thr, None)
            results[hwd][area_name]["thr_entry"] = max_entry

    areas = list(area_alias.values())
    hwds = list(reverse_hwd_alias.keys())

    # ---- LaTeX output ----
    print(r"\renewcommand{\arraystretch}{2}")
    print(r"\begin{tabular}{|c|" + "c" * (2 * len(hwds)) + "|}")
    print(r"\hline")

    # Header row
    header = ["Area"]
    for hwd in hwds:
        header.append(f"{hwd} Throughput")
        header.append(f"{hwd} Latency")
    print(" & ".join(header) + r" \\")
    print(r"\hline")

    # Area rows
    for area in areas:
        row = [area.replace("-", " ").title()]
        for hwd in hwds:
            # Throughput cell
            thr = results[hwd][area]["thr"]
            entry = results[hwd][area]["thr_entry"]
            if thr and entry:
                row.append(
                    r"\shortstack[b]{"
                    + f"{thr}\\\\{{\\scriptsize @{format_bytes(entry['msg_size'], None)}, "
                    + f"P={entry['num_pairs']}}}"
                    + "}"
                )
            else:
                row.append("--")

            # Latency cell
            lat = results[hwd][area]["lat"]
            entry = results[hwd][area]["lat_entry"]
            if lat and entry:
                row.append(
                    r"\shortstack[b]{"
                    + f"{lat}\\\\{{\\scriptsize @{format_bytes(entry['msg_size'], None)}, "
                    + f"P={entry['num_pairs']}}}"
                    + "}"
                )
            else:
                row.append("--")

        print(" & ".join(row) + r" \\")
        print(r"\hline")

    print(r"\end{tabular}")



if __name__ == "__main__":
    # Argument parsing
    reverse_hwd_alias = {
        "A100": "NVIDIA_A100-SXM4-40GB",
        "5090": "NVIDIA_GeForce_RTX_5090",
        "3050ti": "NVIDIA_GeForce_RTX_3050_Ti_Laptop_GPU",
    }

    hwd_clock_speed = {
        "A100": 1_410_000_000,
        "5090": 2_407_000_000,
        "3050ti": 1_035_000_000,
    }

    area_alias = {
        1: "cross-thread",
        2: "cross-warp",
        3: "cross-block",
        4: "cross-gpu",
        5: "cross-node",
        6: "cross-host",
    }

    # Table header
    header = (
        f"{'Area':<15}"
        f"{'Hardware':<12}"
        f"{'Min Latency':<18}"
        f"{'Latency @ (P, Size)':<22}"
        f"{'Max Throughput (B/s)':<24}"
        f"{'Throughput @ (P, Size)':<26}"
        f"{'Instructions (/s)':<20}"
        f"{'Extrapolated Throughput (B/s)':<24}"
    )
    print(header)
    print("-" * len(header))

    for area_id in area_alias:
        dir_name = f"../{area_alias[area_id]}/data"
        for hwd in reverse_hwd_alias:
            json_name = f"{dir_name}/{reverse_hwd_alias[hwd]}_metrics.json"
            if not os.path.exists(json_name):
                continue

            with open(json_name, "r") as f:
                json_data = json.load(f)

            min_lat, min_entry = min_latency(json_data)
            max_thr, max_entry = max_throughput(json_data)

            min_lat_loc = f"(P={min_entry['num_pairs']}, {format_bytes(min_entry['msg_size'], None)})"
            max_thr_loc = f"(P={max_entry['num_pairs']}, {format_bytes(max_entry['msg_size'], None)})"

            inst_per_sec = max_entry['msg_size'] / (max_entry['num_pairs'] * max_entry['metrics']['round_trip_latency_avg'] * 1e-9)
            inst_per_sec = 4 * inst_per_sec if area_id != 1 else inst_per_sec
            extrapolated_thr = (hwd_clock_speed[hwd] / inst_per_sec) * max_thr

            print(
                f"{area_alias[area_id]:<15}"
                f"{hwd:<12}"
                f"{time_format(min_lat, None):<18}"
                f"{min_lat_loc:<22}"
                f"{format_bytes(max_thr, None):<24}"
                f"{max_thr_loc:<26}"
                f"{format_instruction(inst_per_sec, None) if area_id != 6 else 'N/A':<20}"
                f"{format_bytes(extrapolated_thr, None) if area_id != 6 else 'N/A':<24}"
            )
    print("\n\nLaTeX Table:")

    # generate_latex_table_with_details(area_alias, reverse_hwd_alias)
    generate_latex_table_with_details_transposed(area_alias, reverse_hwd_alias)
