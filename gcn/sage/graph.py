import oneflow as flow
import oneflow.nn as nn
import flowvision
import flowvision.transforms as transforms

BATCH_SIZE=64
EPOCH_NUM = 1

DEVICE = "cuda" if flow.cuda.is_available() else "cpu"
print("Using {} device".format(DEVICE))

training_data = flowvision.datasets.CIFAR10(
    root="data",
    train=True,
    transform=transforms.ToTensor(),
    download=True,
    source_url="https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/cifar/cifar-10-python.tar.gz",
)

train_dataloader = flow.utils.data.DataLoader(
    training_data, BATCH_SIZE, shuffle=True
)

model = flowvision.models.mobilenet_v2().to(DEVICE)
model.classifer = nn.Sequential(nn.Dropout(0.2), nn.Linear(model.last_channel, 10))
model.train()

loss_fn = nn.CrossEntropyLoss().to(DEVICE)
optimizer = flow.optim.SGD(model.parameters(), lr=1e-3)


for t in range(EPOCH_NUM):
    print(f"Epoch {t+1}\n-------------------------------")
    size = len(train_dataloader.dataset)
    for batch, (x, y) in enumerate(train_dataloader):
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        +
        # Compute prediction error
        pred = model(x)
        loss = loss_fn(pred, y)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        +
        current = batch * BATCH_SIZE
        if batch % 5 == 0:
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
#############################3
class GraphMobileNetV2(flow.nn.Graph):
    def __init__(self):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.add_optimizer(optimizer)

    def build(self, x, y):
        y_pred = self.model(x)
        loss = self.loss_fn(y_pred, y)
        loss.backward()
        return loss


graph_mobile_net_v2 = GraphMobileNetV2()
# graph_mobile_net_v2.debug()

for t in range(EPOCH_NUM):
    print(f"Epoch {t+1}\n-------------------------------")
    size = len(train_dataloader.dataset)
    for batch, (x, y) in enumerate(train_dataloader):
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        loss = graph_mobile_net_v2(x, y)
        current = batch * BATCH_SIZE
        if batch % 5 == 0:
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")