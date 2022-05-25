# RUN: python3 %s | FileCheck %s
import oneflow as flow
from oneflow_iree.compiler import Runner


class RELU(flow.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = flow.nn.ReLU()

    def forward(self, x):
        return self.relu(x)

class GraphModule(flow.nn.Graph):
    def __init__(self):
        super().__init__()
        self.fw = RELU()

    def build(self, x):
        return self.fw(x)


func = Runner(GraphModule).cuda()


# CHECK:[0. 1.]
input = flow.Tensor([-1, 1.])
output = func(input)
print(output)

# CHECK:[1. 0.]
input = flow.Tensor([1, -1.])
output = func(input)
print(output)

# CHECK:[0. 1. 0.]
func = func.cpu()
input = flow.Tensor([-1, 1., -2])
output = func(input)
print(output)

# CHECK:[0. 1.]
input = flow.Tensor([-1, 1.])
output = func(input)
print(output)