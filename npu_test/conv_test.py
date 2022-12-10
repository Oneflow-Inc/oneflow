import oneflow as flow
import oneflow.nn as nn
import time
flow.manual_seed(1)
use_cpu = 0
if use_cpu:
    inputs = flow.randn(2,3, 1024, 1024, requires_grad = True, dtype=flow.float32)
    print(inputs.sum())
    filters = flow.randn(6, 3, 3, 3, requires_grad = True,dtype=flow.float32)
    out = nn.functional.conv2d(inputs, filters, stride = [1,1], padding = [1,1],dilation=[1,1],channel_pos="channels_first")
    print(out.shape)
    l = out.sum()
    l.backward()
    print(filters.grad)
else:
    inputs_g = flow.randn(2, 3, 1024, 1024,requires_grad = False, dtype=flow.float32)
    inputs = inputs_g.to("npu").to(flow.float16)

    filters_g = flow.randn(6, 3, 3, 3, requires_grad = True,dtype=flow.float32)
    filters = filters_g.to("npu").to(flow.float16)

    out = nn.functional.conv2d(inputs, filters, stride = [1,1], padding = [1,1],dilation=[1,1],channel_pos="channels_first")
    # print(out)
    l = out.sum()
    l.backward()
    print("grad:",filters_g.grad)