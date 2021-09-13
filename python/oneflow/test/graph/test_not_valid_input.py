import oneflow as flow

class MyGraph(flow.nn.Graph):
    def __init__(self):
        super().__init__()
        self.module = flow.nn.ReLU()

    def build(self, x):
        out = self.module(x)
        return out

my_g = MyGraph()

x = flow.tensor(0.5)
out = my_g(x)
print("scalar out numpy ", out.numpy())

B = [flow.sbp.broadcast]
P0 = flow.placement("cuda", {0: [0]})
new_consistent_x = flow.Tensor([1.0, 2.0]).to_consistent(placement=P0, sbp=B)
print("consistent input ", new_consistent_x)
out1 = my_g(new_consistent_x)  # ERROR! 
print("consistent out numpy ", out.numpy())
