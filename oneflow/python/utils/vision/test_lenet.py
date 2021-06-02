import oneflow.experimental as flow
import oneflow.experimental.nn as nn

import transforms as transforms
import oneflow.python.utils.data as data
from datasets.mnist import FashionMNIST
import os
import time

device = flow.device("cuda")

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
        # print("img.shape >>>>>>>>>>>> ", img.shape)
        feature = self.conv(img)
        output = self.fc(feature.reshape(shape=[img.shape[0], -1]))
        return output


net = LeNet()
net.to(flow.device('cuda'))
print(net)

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


batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size=batch_size)

# 本函数已保存在d2lzh_pytorch包中方便以后使用。该函数将被逐步改进。
def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, nn.Module):
        # 如果没指定device就使用net的device
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    with flow.no_grad():
        for X, y in data_iter:
            X = X.unsqueeze(dim=1)
            if isinstance(net, nn.Module):
                net.eval() # 评估模式, 这会关闭dropout
                print("(net(X.to(device)).argmax(dim=1).numpy() == y.to(device).numpy()).sum() >>>>> ", (net(X.to(device)).argmax(dim=1).numpy() == y.to(device).numpy()).sum())
                acc_sum += (net(X.to(device)).argmax(dim=1).numpy() == y.to(device).numpy()).sum()
                net.train() # 改回训练模式
            else: # 自定义的模型, 3.13节之后不会用到, 不考虑GPU
                if('is_training' in net.__code__.co_varnames): # 如果有is_training这个参数
                    # 将is_training设置成False
                    acc_sum += (net(X, is_training=False).argmax(dim=1).numpy() == y.numpy()).float().sum()
                else:
                    acc_sum += (net(X).argmax(dim=1).numpy() == y.numpy()).float().sum()
            n += y.shape[0]
    return acc_sum / n


# 本函数已保存在d2lzh_pytorch包中方便以后使用
def train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs):
    net = net.to(device)
    print("training on ", device)
    loss = nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        for X, y in train_iter:
            X = X.unsqueeze(dim=1) # NOTE:image shape of dataloader should be flow.Size([256, 1, 28, 28])
            X = X.to(device)
            X.requires_grad=True
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_l_sum += l.numpy()
            train_acc_sum += (y_hat.argmax(dim=1).numpy() == y.numpy()).sum()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))
            

lr, num_epochs = 0.001, 5
optimizer = flow.optim.Adam(net.parameters(), lr=lr)
train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)

