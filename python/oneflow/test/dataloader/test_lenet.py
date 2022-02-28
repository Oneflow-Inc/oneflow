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
import unittest

import oneflow as flow
import oneflow.nn as nn
import oneflow.unittest

from data_utils import load_data_fashion_mnist


# reference: http://tangshusen.me/Dive-into-DL-PyTorch/#/chapter05_CNN/5.5_lenet
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),  # in_channels, out_channels, kernel_size
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # kernel_size, stride
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 4 * 4, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
        )

    def forward(self, img):
        feature = self.conv(img)
        feature = feature.flatten(start_dim=1)
        output = self.fc(feature)
        return output


def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, nn.Module):
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    net.eval()
    with flow.no_grad():
        for X, y in data_iter:
            X = X.to(device=device)
            y = y.to(device=device)
            acc_sum += (net(X).argmax(dim=1).numpy() == y.numpy()).sum()
            n += y.shape[0]
    net.train()
    return acc_sum / n


def _test_train_and_eval(test_case):
    if os.getenv("ONEFLOW_TEST_CPU_ONLY"):
        device = flow.device("cpu")
    else:
        device = flow.device("cuda")
    net = LeNet()
    lr, num_epochs = 0.02, 1
    optimizer = flow.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    net.to(device)

    batch_size = 256
    data_dir = os.path.join(
        os.getenv("ONEFLOW_TEST_CACHE_DIR", "./data-test"), "fashion-mnist-lenet"
    )
    source_url = "https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/mnist/Fashion-MNIST/"

    train_iter, test_iter = load_data_fashion_mnist(
        batch_size=batch_size,
        resize=None,
        root=data_dir,
        download=True,
        source_url=source_url,
        num_workers=0,
    )
    loss = nn.CrossEntropyLoss()
    loss.to(device)

    final_accuracy = 0

    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device=device)
            y = y.to(device=device)
            # forward
            y_hat = net(X)
            l = loss(y_hat, y).sum()
            # backward
            l.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_l_sum += l.numpy()
            train_acc_sum += (y_hat.argmax(dim=1).numpy() == y.numpy()).sum()
            n += y.shape[0]
            batch_count += 1
            if batch_count == 20:
                break

        test_acc = evaluate_accuracy(test_iter, net)
        final_accuracy = train_acc_sum / n
        print(
            "epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec"
            % (
                epoch + 1,
                train_l_sum / batch_count,
                final_accuracy,
                test_acc,
                time.time() - start,
            )
        )
    # test_case.assertLess(0.4, final_accuracy)


@flow.unittest.skip_unless_1n1d()
class TestLenet(flow.unittest.TestCase):
    def test_lenet(test_case):
        _test_train_and_eval(test_case)


if __name__ == "__main__":
    unittest.main()
