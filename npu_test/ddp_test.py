import oneflow as flow
from oneflow.nn.parallel import DistributedDataParallel as ddp

# train_x = [
#     flow.tensor([[1, 2], [2, 3]], dtype=flow.float32),
#     flow.tensor([[4, 6], [3, 1]], dtype=flow.float32),
# ]
# train_y = [
#     flow.tensor([[8], [13]], dtype=flow.float32),
#     flow.tensor([[26], [9]], dtype=flow.float32),
# ]
train_x = flow.tensor([[1, 3], [2, 3]], dtype=flow.float32)
train_y = flow.tensor([[8], [13]], dtype=flow.float32)
class Model(flow.nn.Module):
    def __init__(self):
        super().__init__()
        self.lr = 0.01
        self.iter_count = 500
        self.w = flow.nn.Parameter(flow.tensor([[0], [0]], dtype=flow.float32))

    def forward(self, x):
        x = flow.matmul(x, self.w)
        return x

use_npu = False

m = Model()
if use_npu:
    m = m.to("npu")
m = ddp(m)
loss = flow.nn.MSELoss(reduction="sum")
if use_npu:
   optimizer = flow.optim.TORCH_SGD(m.parameters(), m.lr)  
else:
   optimizer = flow.optim.SGD(m.parameters(), m.lr)

for i in range(0, 1):
    rank = flow.env.get_rank()
    print(rank)
    x = train_x
    y = train_y
    if use_npu:
        x = x.to("npu")
        y = y.to("npu")
    y_pred = m(x)
    l = y_pred.sum()
    if (i + 1) % 50 == 0:
        print(f"{i+1}/{m.iter_count} loss:{l}")

    optimizer.zero_grad()
    l.backward()
    optimizer.step()

print(f"\nw:{m.w}")