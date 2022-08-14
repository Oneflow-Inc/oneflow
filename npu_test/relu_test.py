import oneflow as flow
import oneflow.nn as nn
import time
use_cpu = False
if use_cpu:
    inputs = flow.tensor([1.,-2.,3.],requires_grad=True, dtype=flow.float32)
    out = flow.nn.functional.relu(inputs)
    l = out.sum()
    l.backward()
    print(inputs.grad)
else:
    inputs = flow.tensor([1.,-2.,3.],requires_grad=True, dtype=flow.float16).to("npu")
    out = flow.nn.functional.relu(inputs)
    # l = out.sum()
    # l.backward()
