import os
import time

import oneflow.experimental as flow
import oneflow.experimental.nn as nn
import transforms as transforms
import oneflow.python.utils.data as data
from datasets.mnist import FashionMNIST


# ref: http://tangshusen.me/Dive-into-DL-PyTorch/#/chapter05_CNN/5.5_lenet
flow.enable_eager_execution()

def load_data_fashion_mnist(batch_size, resize=None, root='./test/FashionMNIST'):
    """Download the Fashion-MNIST dataset and then load into memory."""
    root = os.path.expanduser(root)
    trans = []
    if resize:
        trans.append(transforms.Resize(size=resize))
    trans.append(transforms.ToTensor())
    transform = transforms.Compose(trans)

    mnist_train = FashionMNIST(root=root, train=True, transform=transform, download=True)
    mnist_test = FashionMNIST(root=root, train=False, transform=transform, download=True)
    num_workers = 0

    train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=num_workers)
    test_iter = data.DataLoader(mnist_test, batch_size, shuffle=False, num_workers=num_workers)
    return train_iter, test_iter


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, 5), # in_channels, out_channels, kernel_size
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2), # kernel_size, stride
            nn.Conv2d(6, 16, 5),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(16*4*4, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, 10)
        )

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.reshape(shape=[img.shape[0], -1]))
        return output

# define LeNet module
class LeNet5(nn.Module):
    def __init__(self, n_classes):
        super(LeNet5, self).__init__()
        self.feature_extractor = nn.Sequential(            
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
            nn.Tanh()
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=n_classes),
        )
        self.m1 = nn.Linear(in_features=120, out_features=84)
        self.m2 = nn.Tanh()
        self.m3 = nn.Linear(in_features=84, out_features=n_classes)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = flow.flatten(x, 1)
        logits = self.classifier(x)
        probs = flow.softmax(logits, dim=1)
        return logits


# network = LeNet()
network = LeNet5(n_classes=10)

for params in network.parameters():
    nn.init.normal_(params, mean=0, std=0.01)

device = flow.device("cuda") # segmentfault in cpu mode
network.to(device)


batch_size = 128
train_iter, test_iter = load_data_fashion_mnist(batch_size=batch_size, resize=32)
loss = nn.CrossEntropyLoss()
loss.to(device)

lr, num_epochs = 0.001, 10
optimizer = flow.optim.Adam(network.parameters(), lr=lr)


# 本函数已保存在d2lzh_pytorch包中方便以后使用。该函数将被逐步改进。
def evaluate_accuracy(net, device, data_iter):
    if device is None and isinstance(net, nn.Module):
        # 如果没指定device就使用net的device
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    with flow.no_grad():
        for X, y in data_iter:
            X = X.to(device=device, dtype=flow.float32)
            y = y.to(device=device, dtype=flow.int64)
            if isinstance(net, nn.Module):
                net.eval() # 评估模式, 这会关闭dropout 
                acc_sum += (net(X).argmax(dim=1).numpy() == y.numpy()).sum()
                net.train() # 改回训练模式
            else: # 自定义的模型, 3.13节之后不会用到, 不考虑GPU
                if('is_training' in net.__code__.co_varnames): # 如果有is_training这个参数
                    # 将is_training设置成False
                    acc_sum += (net(X, is_training=False).argmax(dim=1).numpy() == y.numpy()).sum()
                else:
                    acc_sum += (net(X).argmax(dim=1).numpy() == y.numpy()).sum()
            n += y.shape[0]
    return acc_sum / n


# 本函数已保存在d2lzh_pytorch包中方便以后使用
def train_ch5(net, device, train_iter, test_iter, num_epochs):
    net = net.to(device)
    print("training on ", device)
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        for X, y in train_iter:
            X.requires_grad=True
            X = X.to(device=device, dtype=flow.float32)
            y = y.to(device=device, dtype=flow.int64)
            y_hat = net(X)
            l = loss(y_hat, y)

            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            train_l_sum += l.numpy()
            train_acc_sum += (y_hat.argmax(dim=1).numpy() == y.numpy()).sum()
            n += y.shape[0]
            batch_count += 1

        test_acc = evaluate_accuracy(net, device, test_iter)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))
            

train_ch5(network, device, train_iter, test_iter, num_epochs)

