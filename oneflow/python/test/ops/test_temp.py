import numpy as np
import unittest
import oneflow.experimental as flow
from oneflow.python.nn.parameter import Parameter
flow.enable_eager_execution(True)
init_val = np.random.randn(2, 3)
class CustomModule(flow.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = Parameter(flow.Tensor(init_val))
    def forward(self, x):
        return x + self.w
m = CustomModule()
inputs = flow.ones((2, 3))
out = m(inputs)
s = flow.sum(out)
print(out.numpy())
print(s.numpy())
s.backward()
print(m.w.grad)
print(m.w.grad.numpy())
