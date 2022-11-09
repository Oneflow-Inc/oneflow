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
import unittest
import time

import oneflow.unittest
import oneflow as flow
import oneflow.nn as nn
from data_utils import load_data_fashion_mnist


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


class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self, x):  # x shape: (batch, *, *, ...)
        res = x.reshape(x.shape[0], -1)
        return res


def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, nn.Module):
        # using net device if not specified
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    net.eval()
    with flow.no_grad():
        for X, y in data_iter:
            X = X.to(device=device)
            y = y.to(device=device)
            acc_sum += (
                net(X.to(device)).argmax(dim=1).numpy() == y.to(device).numpy()
            ).sum()
            n += y.shape[0]
    net.train()
    return acc_sum / n


def _test(test_case):
    num_inputs, num_outputs, num_hiddens = 784, 10, 256
    net = nn.Sequential(
        FlattenLayer(),
        nn.Linear(num_inputs, num_hiddens),
        nn.ReLU(),
        nn.Linear(num_hiddens, num_outputs),
    )

    if os.getenv("ONEFLOW_TEST_CPU_ONLY"):
        device = flow.device("cpu")
    else:
        device = flow.device("cuda")
    net.to(device)

    batch_size = 256
    num_epochs = 1
    data_dir = os.path.join(
        os.getenv("ONEFLOW_TEST_CACHE_DIR", "./data-test"), "fashion-mnist"
    )
    source_url = "https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/mnist/Fashion-MNIST/"
    train_iter, test_iter = load_data_fashion_mnist(
        batch_size, resize=None, root=data_dir, download=True, source_url=source_url
    )
    loss = nn.CrossEntropyLoss()
    loss.to(device)

    optimizer = flow.optim.SGD(net.parameters(), lr=0.1)
    final_accuracy = 0
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        start = time.time()
        for X, y in train_iter:
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
            if n > 200:
                break

        test_acc = evaluate_accuracy(test_iter, net)
        final_accuracy = train_acc_sum / n
        print(
            "epoch %d, loss %.4f, train acc %.3f, test acc %.3f, cost >>>>>>> %s(s)"
            % (
                epoch + 1,
                train_l_sum / n,
                final_accuracy,
                test_acc,
                str(time.time() - start),
            )
        )
        final_accuracy = train_acc_sum / n
    # test_case.assertLess(0.60, final_accuracy)


@flow.unittest.skip_unless_1n1d()
class TestFashionMnistDataset(flow.unittest.TestCase):
    def test_fashion_mnist_dataset(test_case):
        _test(test_case)


if __name__ == "__main__":
    unittest.main()
