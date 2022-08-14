import oneflow as flow
import oneflow.nn as nn
import time
flow.manual_seed(1)
use_cpu = 1
if use_cpu:
    dtype = flow.float32
else:
    dtype = flow.float32
if use_cpu:
    inputs = flow.randn(2,64,requires_grad=True, dtype=dtype)
    weights = flow.randn(1,64,requires_grad=True, dtype=dtype)
    out = flow.nn.functional.linear(inputs,weights)
    l = out.sum()
    l.backward()
    print(weights.grad)
else:
    inputs = flow.randn(2, 64, requires_grad=True, dtype=dtype).to("npu")
    weights = flow.randn(1, 64, requires_grad=True, dtype=dtype).to("npu")
    out = flow.nn.functional.linear(inputs, weights)
    l = out.sum()
    l.backward()
