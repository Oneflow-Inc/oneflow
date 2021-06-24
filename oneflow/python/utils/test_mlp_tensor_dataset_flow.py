import numpy as np
import oneflow.experimental as flow
import oneflow.python.utils.data as Data
import oneflow.experimental.nn as nn


flow.enable_eager_execution()


num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = flow.tensor(np.random.normal(0, 1, (num_examples, num_inputs)), dtype=flow.float)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += flow.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=flow.float)


batch_size = 10
# 将训练数据的特征和标签组合
dataset = Data.TensorDataset(features, labels)
# 随机读取小批量
data_iter = Data.DataLoader(dataset, batch_size, shuffle=True, num_workers=2)


class MSELoss(nn.Module):
    def __init__(
        self,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        assert reduction in [
            "none",
            "mean",
            "sum",
        ], "{} is not a valid value for reduction, The reduction must be the one of `none`, `mean`, `sum`. ".format(
            reduction
        )
        self.reduction = reduction

    def forward(self, input, target):
        assert (
            input.shape == target.shape
        ), "The Input shape must be the same as Target shape"
        squared_difference = flow.square(flow.sub(input, target))

        if self.reduction == "mean":
            return flow.mean(squared_difference)
        elif self.reduction == "sum":
            return flow.sum(squared_difference)
        else:
            return squared_difference

for X, y in data_iter:
    print(X, y)
    break


class LinearNet(nn.Module):
    def __init__(self, n_feature):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(n_feature, 1)
    # forward 定义前向传播
    def forward(self, x):
        y = self.linear(x)
        return y


net = LinearNet(num_inputs)
print(net) # 使用print可以打印出网络的结构


flow.nn.init.normal_(net.linear.weight, mean=0, std=0.01)
flow.nn.init.constant_(net.linear.bias, val=0)  # 也可以直接修改bias的data: net[0].bias.data.fill_(0)


loss = MSELoss()

optimizer = flow.optim.SGD(net.parameters(), lr=0.03)
print(optimizer)


num_epochs = 10
for epoch in range(1, num_epochs + 1):
    for X, y in data_iter:
        output = net(X)
        l = loss(output, y)
        optimizer.zero_grad() # 梯度清零，等价于net.zero_grad()
        l.backward()
        optimizer.step()
    print('epoch %d, loss: %f' % (epoch, l.numpy()))
