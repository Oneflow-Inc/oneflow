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

import oneflow.unittest
import oneflow as flow
import torch
import torch.nn.functional as F
import torchvision as vision
import torch.nn as nn
import torch.optim as optim


classes = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def test(test_case):
    if os.getenv("ONEFLOW_TEST_CPU_ONLY"):
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
    net = Net()
    net.to(device)

    optimizer = optim.SGD(net.parameters(), lr=0.002, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    criterion.to(device)

    transform = vision.transforms.Compose(
        [
            vision.transforms.ToTensor(),
            vision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    train_epoch = 10
    batch_size = 4
    data_dir = os.path.join(
        os.getenv("ONEFLOW_TEST_CACHE_DIR", "./data-test-torch"), "cifar10"
    )

    trainset = vision.datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    final_loss = 0
    for epoch in range(1, train_epoch + 1):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 1):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(dtype=torch.float32, device=device)
            labels = labels.to(dtype=torch.int64, device=device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.cpu().detach().numpy()
            if i % 200 == 0:  # print every 200 mini-batches
                final_loss = running_loss / 200
                print("epoch: %d  step: %5d  loss: %.3f " % (epoch, i, final_loss))
                running_loss = 0.0
                break

    print("final loss : ", final_loss)
    # test_case.assertLess(final_loss, 1.50)


@flow.unittest.skip_unless_1n1d()
class TestCifarDataset(flow.unittest.TestCase):
    def test_cifar_dataset(test_case):
        test(test_case)


if __name__ == "__main__":
    unittest.main()
    # 1 epoch training log
    # epoch: 1  step:  2000  loss: 2.107
    # epoch: 1  step:  4000  loss: 1.838
    # epoch: 1  step:  6000  loss: 1.644
    # epoch: 1  step:  8000  loss: 1.535
    # epoch: 1  step: 10000  loss: 1.528
    # epoch: 1  step: 12000  loss: 1.476
