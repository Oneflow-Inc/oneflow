import oneflow as flow
import numpy as np
import time
print_control = 1
time_start = 0
def start_tax(str):
    if not print_control:
        return
    global time_start
    flow.npu.synchronize()
    time_start = time.time()
    print(str)

def end_tax(str):
    if not print_control:
        return
    global time_start
    flow.npu.synchronize()
    print(str," ",time.time()-time_start) 
    time_start = time.time()

np.random.seed(1)
use_npu = 1
inp = np.random.randn(256,64,112,112)
model = flow.nn.MaxPool2d(kernel_size=3, stride=2, padding=1,return_indices=True)
# model = flow.nn.Conv2d(64,64,kernel_size=3,stride=2,padding=1,bias=False)
inputs = flow.tensor(inp,dtype=flow.float32)
# out, indices = model(inputs)
# l = out.sum()
# l.backward()
# print(inputs.grad)
if use_npu:
    model_n = model.to("npu:0").to(flow.float16)
    start = time.time()
    for i in range(10):
        start_tax(f"step==========>{i}")
        inputs_n = inputs.to("npu:0").to(flow.float16)
        inputs_n.requires_grad_()
        end_tax('copy')
        out, indices= model_n(inputs_n)
        #out = inputs_n + inputs_n
        l = out.sum()
        # back = flow.ones_like(out)
        end_tax('forward')
        l.backward()
        #out.backward(back)
        end_tax('backward')
    print("all time: ",time.time()-start)
# ones_like
# broadcast_like