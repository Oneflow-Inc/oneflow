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

import flowvision as vision
import flowvision.transforms as transforms

import oneflow.unittest
import oneflow as flow
import oneflow.nn as nn
from data_utils import load_data_mnist


data_dir = os.path.join(
    os.getenv("ONEFLOW_TEST_CACHE_DIR", "./data-test"), "mnist-dataset"
)
train_iter, test_iter = load_data_mnist(
    batch_size=128,
    download=True,
    root=data_dir,
    source_url="https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/mnist/MNIST/",
)


def evaluate_accuracy(data_iter, net, device=None):
    n_correct, n_samples = 0.0, 0
    net.to(device)
    net.eval()
    with flow.no_grad():
        for images, labels in data_iter:
            images = images.reshape(-1, 28 * 28)
            images = images.to(device=device)
            labels = labels.to(device=device)
            n_correct += (net(images).argmax(dim=1).numpy() == labels.numpy()).sum()
            n_samples += images.shape[0]
    net.train()
    return n_correct / n_samples


class Net(nn.Module):
    def __init__(
        self, input_size=784, hidden_size1=128, hidden_size2=64, num_classes=10
    ):
        super(Net, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.l2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.l3 = nn.Linear(hidden_size2, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu1(out)
        out = self.l2(out)
        out = self.relu2(out)
        out = self.l3(out)
        return out


def _test_train_and_eval(test_case):
    if os.getenv("ONEFLOW_TEST_CPU_ONLY"):
        device = flow.device("cpu")
    else:
        device = flow.device("cuda")

    model = Net()
    model.to(device)

    loss = nn.CrossEntropyLoss().to(device)
    optimizer = flow.optim.SGD(model.parameters(), lr=0.10)

    num_epochs = 1
    for epoch in range(num_epochs):
        train_loss, n_correct, n_samples = 0.0, 0.0, 0
        for images, labels in train_iter:
            images = images.reshape(-1, 28 * 28)
            images = images.to(device=device)
            labels = labels.to(device=device)
            features = model(images)
            l = loss(features, labels).sum()
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            train_loss += l.numpy()
            n_correct += (features.argmax(dim=1).numpy() == labels.numpy()).sum()
            n_samples += images.shape[0]
            if n_samples > 2000:
                break

        test_acc = evaluate_accuracy(test_iter, model, device)
        train_acc = n_correct / n_samples
        print(
            "epoch %d, train loss %.4f, train acc %.3f, test acc %.3f"
            % (epoch + 1, train_loss / n_samples, train_acc, test_acc)
        )
        # test_case.assertLess(0.8, test_acc)


@flow.unittest.skip_unless_1n1d()
class TestMnistDataset(flow.unittest.TestCase):
    def test_mnist_dataset(test_case):
        _test_train_and_eval(test_case)


if __name__ == "__main__":
    unittest.main()
