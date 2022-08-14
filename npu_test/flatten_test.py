import oneflow as flow
import oneflow.nn as nn
import time
use_cpu = 0

dtype = flow.float32
if use_cpu:
    m = flow.nn.Flatten()
    x = flow.ones(1, 4, 4, 4, dtype=dtype)
    y = m(x)
    print(y)
else:
    m = flow.nn.Flatten().to("npu")
    x = flow.ones(1, 4, 4, 4, dtype=dtype).to("npu")
    x.requires_grad = True
    y = m(x)
    l = y.sum()
    l.backward()
    print(y)