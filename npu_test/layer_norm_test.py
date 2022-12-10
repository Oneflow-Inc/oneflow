import oneflow
import numpy as np
np.random.seed(0)
x = np.random.randn(2,3,784)

input = oneflow.tensor(x,dtype=oneflow.float32,requires_grad=True).to('npu').to(oneflow.half)
ln = oneflow.nn.LayerNorm(784).to('npu').to(oneflow.half)
output = ln(input)
print(output)