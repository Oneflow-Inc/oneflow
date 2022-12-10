import oneflow as flow
import oneflow.nn as nn
import numpy as np
bn = nn.BatchNorm2d(num_features=3,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).to("npu")
bn.train()
optimizer = flow.optim.TORCH_SGD(bn.parameters(), lr = 1., momentum=0.9)
for i in range(3):
    np.random.seed(2)
    input_np = np.random.randn(1,3,3,3)
    inputs = flow.tensor(input_np,requires_grad=True,dtype=flow.float32)
    inputs_n = inputs.to("npu")
    print(bn.weight)
    out = bn(inputs_n)
    out = out * 102400.0
    out = out.sum()
    out.backward()
    print(bn.weight.grad)
    optimizer.step()
    print(bn.weight)
    print('-'*10)
