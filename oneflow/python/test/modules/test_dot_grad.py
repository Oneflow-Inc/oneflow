import torch
import numpy as np
# import oneflow
# dx, dy = oneflow.F.dot_grad(oneflow.tensor([1.2,2.3]), oneflow.tensor([3.0, 4.0]), oneflow.tensor([1.0]))

# print(dx, dy)

# x = np.random.randn(2)
# y = np.random.randn(2)

# print(x, y)

# out  = np.dot(x, y)

x = torch.tensor([2.3, 2.6],requires_grad=True)
y = torch.tensor([3.7, 8.6],requires_grad=True)


z = torch.dot(x, y)

z.backward()
print(x.grad)
print(y.grad)
