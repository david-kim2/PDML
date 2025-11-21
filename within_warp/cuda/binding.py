import torch
from torch.utils.cpp_extension import load
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

warp_echo_lib = load(
    name="warp_echo_lib",
    sources=[os.path.join(current_dir, "warp_echo.cu")],
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    verbose=True
)

def warp_echo_forward(message, message_size_bytes):
    """
    Python wrapper for our warp_echo_forward CUDA kernel.
    
    Returns:
        torch.Tensor: [6] array with timestamps
                     [cl_tx_st, cl_tx_ed, cl_rx, sv_rx, sv_tx_st, sv_tx_ed]
    """
    return warp_echo_lib.warp_echo_forward(message, message_size_bytes)