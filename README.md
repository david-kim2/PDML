# PDML

- `src/cross-thread`: Threads communicating with each other within a warp.
- `src/cross-warp`: Threads communicating across warps within a block.
- `src/cross-block`: Threads communicating across blocks within a GPU.
- `src/cross-gpu`: Threads communicating across GPUs within a node.
- `src/cross-gpu`: Threads communicating across GPUs across nodes.
- `src/cross-host`: Threads communicating between GPU and host CPU.
