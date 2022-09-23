# OneFlow extension of MLIR features]

## KernelLaunchOp

### Stage 1

- 1:1 conversion from user op to kernel launch op

### Stage 2

- multi user op merged into one single kernel launch op

### Stage 3

- oneflow-opt and similar non-python execution environment
- multi-gpu/multi-node compilation support (in the beginning it is all single-node with broadcast SBP signature)

### relationship with MlirJitOp

- the graph of a MlirJitOp might contain one or multiple kernel launch op
- an op inside the graph of MlirJitOp could be optionally lowered to a kernel launch op
