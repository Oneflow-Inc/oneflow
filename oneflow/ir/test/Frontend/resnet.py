import oneflow as flow
from flowvision.models import resnet50
print(flow.__file__)

from oneflow.nn.graph.compiler import Runner, Dev

class Graph(flow.nn.Graph):
    def __init__(self):
        super().__init__()
        self.model = resnet50()

    def build(self, x):
        return self.model(x)


graph_to_run = Graph()
func = Dev(graph_to_run)
# func = Runner(graph_to_run)
input = flow.zeros([1, 3, 224, 224])
output = func(input)

# print(output)