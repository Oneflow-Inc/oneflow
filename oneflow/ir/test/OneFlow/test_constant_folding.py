import oneflow as flow
import oneflow.nn as nn
import os

os.environ["ONEFLOW_MLIR_ENABLE_ROUND_TRIP"] = '1'
os.environ["ONEFLOW_MLIR_ENABLE_CODEGEN_FUSERS"] = '1'
os.environ["ONEFLOW_MLIR_STDOUT"] = '1'


class FooModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.x = nn.Parameter(flow.tensor([1, 1]), False)
        self.y = nn.Parameter(flow.tensor([[1, 2], [3, 4]]), False)

    def forward(self):
        return self.x + self.y


class AddTwoGraph(nn.Graph):
    def __init__(self):
        super().__init__()
        self.model = FooModel()
        flow.save(self.model.state_dict(), './test_model')

    def build(self):
        return self.model()


if __name__ == '__main__':
    graph = AddTwoGraph()
    print(graph)
    print(graph())
