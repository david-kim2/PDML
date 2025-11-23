import matplotlib.pyplot as plt
import numpy as np
import json
import os

# fabric times: cl_rx_ed-cl_rx_st, sv_rx_ed-sv_rx_st
# single trip latencies: sv_rx_ed-cl_tx_st, cl_rx_ed-sv_tx_st
# round trip latencies: cl_rx_ed-cl_tx_st, sv_rx_ed-sv_tx_st
# sending delays: cl_tx_ed-cl_tx_st, sv_tx_ed-sv_tx_st

def compute_metrics(json_path):
    # "message_size": 1,
    # "n_runs": 10,
    # "num_pairs": 1,
    # "run0": {
    #     "pair0": {
    #         "client_recv_end": 1763861517457104992,
    #         "client_recv_start": 1763861517457104928,
    #         "client_trans_end": 1763861517457104864,
    #         "client_trans_start": 1763861517457104832,
    #         "server_recv_end": 1763861517457104896,
    #         "server_recv_start": 1763861517457104864,
    #         "server_trans_end": 1763861517457104960,
    #         "server_trans_start": 1763861517457104896
    #     }
    # },

    with open(json_path, 'r') as f:
        data = json.load(f)

    metrics = {
        "hdw_trans_delay_client": [0] * data["n_pairs"],
        "hdw_trans_delay_server": [0] * data["n_pairs"],
        "single_trip_latencies_client": [0] * data["n_pairs"],
        "single_trip_latencies_server": [0] * data["n_pairs"],
        "sending_delays_client": [0] * data["n_pairs"],
        "sending_delays_server": [0] * data["n_pairs"],
        "round_trip_latencies": [0] * data["n_pairs"],
    }
    n_runs = data["n_runs"]
    num_pairs = data["num_pairs"]
    for run_idx in range(n_runs):
        run_key = f"run{run_idx}"

        for pair_idx in range(num_pairs):
            pair_key = f"pair{pair_idx}"
            entry = data[run_key][pair_key]

            cl_rx_ed, cl_rx_st = entry["client_recv_end"], entry["client_recv_start"]
            sv_rx_ed, sv_rx_st = entry["server_recv_end"], entry["server_recv_start"]
            cl_tx_ed, cl_tx_st = entry["client_trans_end"], entry["client_trans_start"]
            sv_tx_ed, sv_tx_st = entry["server_trans_end"], entry["server_trans_start"]

            hwd_trans_delay_client = cl_rx_ed - cl_rx_st
            hwd_trans_delay_server = sv_rx_ed - sv_rx_st
            single_trip_latency_client = sv_rx_ed - cl_tx_st
            single_trip_latency_server = cl_rx_ed - sv_tx_st
            round_trip_latency = cl_rx_ed - cl_tx_st
            sending_delay_client = cl_tx_ed - cl_tx_st
            sending_delay_server = sv_tx_ed - sv_tx_st

            metrics["hdw_trans_delay_client"][pair_idx] += hwd_trans_delay_client
            metrics["hdw_trans_delay_server"][pair_idx] += hwd_trans_delay_server
            metrics["single_trip_latencies_client"][pair_idx] += single_trip_latency_client
            metrics["single_trip_latencies_server"][pair_idx] += single_trip_latency_server
            metrics["round_trip_latencies"][pair_idx] += round_trip_latency
            metrics["sending_delays_client"][pair_idx] += sending_delay_client
            metrics["sending_delays_server"][pair_idx] += sending_delay_server

    # Average the metrics
    total_entries = n_runs * num_pairs

if __name__ == "__main__":
    data_dir = "data/"
    # create a category for each folder in data_dir (each folder is a device name)
    categories = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]