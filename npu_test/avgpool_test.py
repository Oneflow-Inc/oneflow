import oneflow as flow
import oneflow.nn as nn
import time
use_npu = 1

dtype = flow.float32

m = flow.nn.AvgPool2d(kernel_size=3, padding=0, stride=1)
x = flow.randn(1, 4, 4, 4, dtype=dtype,requires_grad=True)
y = m(x)
print(y)
if use_npu:
    m = flow.nn.AvgPool2d(kernel_size=3, padding=0, stride=1).to("npu")
    x =x.to("npu")
    y = m(x)
    # l = y.sum()
    # l.backward()