"""
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import os
import time

import oneflow.experimental as flow
import oneflow.experimental.nn as nn

import oneflow.python.utils.data as data
import oneflow.python.utils.vision.datasets as datasets
import oneflow.python.utils.vision.transforms as transforms

flow.enable_eager_execution()


# reference: http://tangshusen.me/Dive-into-DL-PyTorch/#/chapter03_DL-basics/3.10_mlp-pytorch
def load_data_fashion_mnist(batch_size, resize=None, root="./data/fashion-mnist"):
    """Download the Fashion-MNIST dataset and then load into memory."""
    root = os.path.expanduser(root)
    transformer = []
    if resize:
        transformer += [transforms.Resize(resize)]
    transformer += [transforms.ToTensor()]
    transformer = transforms.Compose(transformer)

    mnist_train = datasets.FashionMNIST(
        root=root, train=True, transform=transformer, download=True
    )
    mnist_test = datasets.FashionMNIST(
        root=root, train=False, transform=transformer, download=True
    )
    num_workers = 0
    train_iter = data.DataLoader(
        mnist_train, batch_size, shuffle=True, num_workers=num_workers
    )
    test_iter = data.DataLoader(
        mnist_test, batch_size, shuffle=False, num_workers=num_workers
    )
    return train_iter, test_iter


def get_fashion_mnist_labels(labels):
    """Get text labels for Fashion-MNIST."""
    text_labels = [
        "t-shirt",
        "trouser",
        "pullover",
        "dress",
        "coat",
        "sandal",
        "shirt",
        "sneaker",
        "bag",
        "ankle boot",
    ]
    return [text_labels[int(i)] for i in labels]


num_inputs, num_outputs, num_hiddens = 784, 10, 256

class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self, x):  # x shape: (batch, *, *, ...)
        res = x.reshape(shape=[x.shape[0], -1])
        return res

net = nn.Sequential(
    FlattenLayer(),
    nn.Linear(num_inputs, num_hiddens),
    nn.ReLU(),
    nn.Linear(num_hiddens, num_outputs),
)



device = flow.device("cuda")
net.to(device)


batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size)
loss = nn.CrossEntropyLoss()
loss.to(device)

optimizer = flow.optim.SGD(net.parameters(), lr=0.1)

num_epochs = 10


def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, nn.Module):
        # using net device if not specified
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    with flow.no_grad():
        for X, y in data_iter:
            X = X.to(device=device)
            y = y.to(device=device)
            if isinstance(net, nn.Module):
                net.eval()
                acc_sum += (
                    net(X.to(device)).argmax(dim=1).numpy() == y.to(device).numpy()
                ).sum()
                net.train()
            else:
                if "is_training" in net.__code__.co_varnames:
                    # set is_training=False if has 'is_training' param
                    acc_sum += (
                        net(X, is_training=False).argmax(dim=1).numpy() == y.numpy()
                    ).sum()
                else:
                    acc_sum += (net(X).argmax(dim=1).numpy() == y.numpy()).sum()
            n += y.shape[0]
    return acc_sum / n


def train(
    net,
    train_iter,
    test_iter,
    loss,
    num_epochs,
    batch_size,
    optimizer=None,
):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        start = time.time()
        for X, y in train_iter:
            X.requires_grad = True
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

        test_acc = evaluate_accuracy(test_iter, net)
        print(
            "epoch %d, loss %.4f, train acc %.3f, test acc %.3f, cost >>>>>>> %s(s)"
            % (
                epoch + 1,
                train_l_sum / n,
                train_acc_sum / n,
                test_acc,
                str(time.time() - start),
            )
        )


train(net, train_iter, test_iter, loss, num_epochs, batch_size, optimizer)
