import os
import time

import oneflow.experimental as flow
import oneflow.experimental.nn as nn
import transforms as transforms
import oneflow.python.utils.data as data
from datasets.mnist import FashionMNIST

flow.enable_eager_execution()

# ref: http://tangshusen.me/Dive-into-DL-PyTorch/#/chapter03_DL-basics/3.10_mlp-pytorch

def load_data_fashion_mnist(batch_size, resize=None, root='./test/FashionMNIST'):
    """Download the Fashion-MNIST dataset and then load into memory."""
    root = os.path.expanduser(root)
    transformer = []
    if resize:
        transformer += [transforms.Resize(resize)]
    transformer += [transforms.ToTensor()]
    transformer = transforms.Compose(transformer)

    mnist_train = FashionMNIST(root=root, train=True, transform=transformer, download=True)
    mnist_test = FashionMNIST(root=root, train=False, transform=transformer, download=True)
    num_workers = 0

    train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=num_workers)
    test_iter = data.DataLoader(mnist_test, batch_size, shuffle=False, num_workers=num_workers)
    return train_iter, test_iter

def get_fashion_mnist_labels(labels):
    """Get text labels for Fashion-MNIST."""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


num_inputs, num_outputs, num_hiddens = 784, 10, 256

class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()
    def forward(self, x): # x shape: (batch, *, *, ...)
        res = x.reshape(shape=[x.shape[0], -1])
        return res

net = nn.Sequential(
        FlattenLayer(),
        nn.Linear(num_inputs, num_hiddens),
        nn.ReLU(),
        nn.Linear(num_hiddens, num_outputs), 
        )

for params in net.parameters():
    nn.init.normal_(params, mean=0, std=0.01)


device = flow.device("cuda")
net.to(device)


batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size)
loss = nn.CrossEntropyLoss()
loss.to(device)

optimizer = flow.optim.SGD(net.parameters(), lr=0.1)

num_epochs = 10


# ############################ 5.5 #########################
def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, nn.Module):
        # 如果没指定device就使用net的device
        device = list(net.parameters())[0].device 
    acc_sum, n = 0.0, 0
    with flow.no_grad():
        for X, y in data_iter:
            X = X.to(device=device)
            y = y.to(device=device)
            if isinstance(net, nn.Module):
                net.eval() # 评估模式, 这会关闭dropout
                acc_sum += (net(X.to(device)).argmax(dim=1).numpy() == y.to(device).numpy()).sum()
                net.train() # 改回训练模式
            else: # 自定义的模型, 3.13节之后不会用到, 不考虑GPU
                if('is_training' in net.__code__.co_varnames): # 如果有is_training这个参数
                    # 将is_training设置成False
                    acc_sum += (net(X, is_training=False).argmax(dim=1).numpy() == y.numpy()).sum()
                else:
                    acc_sum += (net(X).argmax(dim=1).numpy() == y.numpy()).sum()
            n += y.shape[0]
    return acc_sum / n


def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
              params=None, lr=None, optimizer=None):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        start = time.time()
        iter_start = start
        for X, y in train_iter:
            t1 = time.time()
            X.requires_grad=True
            X = X.to(device=device)
            y = y.to(device=device)
            y_hat = net(X)
            l = loss(y_hat, y).sum()

            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            train_l_sum += l.numpy()
            train_acc_sum += (y_hat.argmax(dim=1).numpy() == y.numpy()).sum()
            n += y.shape[0]

            t2 = time.time()
            print("train iter >> data prepare cost:", t1-iter_start, "forward iter cost:",t2 - t1)
            iter_start = t2
        
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, cost >>>>>>> %s(s)'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc, str(time.time()-start)))


train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)
