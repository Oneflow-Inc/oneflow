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
import oneflow as flow
import flowvision as vision
import flowvision.transforms as transforms


def load_data_cifar10(
    batch_size,
    data_dir="./data-test/cifar10",
    download=True,
    transform=None,
    source_url=None,
    num_workers=0,
):
    cifar10_train = vision.datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=download,
        transform=transform,
        source_url=source_url,
    )
    cifar10_test = vision.datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=download,
        transform=transform,
        source_url=source_url,
    )

    train_iter = flow.utils.data.DataLoader(
        cifar10_train, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_iter = flow.utils.data.DataLoader(
        cifar10_test, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_iter, test_iter


def load_data_mnist(
    batch_size, resize=None, root="./data/mnist", download=True, source_url=None
):
    """Download the MNIST dataset and then load into memory."""
    root = os.path.expanduser(root)
    transformer = []
    if resize:
        transformer += [transforms.Resize(resize)]
    transformer += [transforms.ToTensor()]
    transformer = transforms.Compose(transformer)

    mnist_train = vision.datasets.MNIST(
        root=root,
        train=True,
        transform=transformer,
        download=download,
        source_url=source_url,
    )
    mnist_test = vision.datasets.MNIST(
        root=root,
        train=False,
        transform=transformer,
        download=download,
        source_url=source_url,
    )
    train_iter = flow.utils.data.DataLoader(
        mnist_train, batch_size, shuffle=True, num_workers=2
    )
    test_iter = flow.utils.data.DataLoader(
        mnist_test, batch_size, shuffle=False, num_workers=2
    )
    return train_iter, test_iter


def get_fashion_mnist_dataset(
    resize=None, root="./data-test/fashion-mnist", download=True, source_url=None,
):
    root = os.path.expanduser(root)
    trans = []
    if resize:
        trans.append(transforms.Resize(resize))
    trans.append(transforms.ToTensor())
    transform = transforms.Compose(trans)

    mnist_train = vision.datasets.FashionMNIST(
        root=root,
        train=True,
        transform=transform,
        download=download,
        source_url=source_url,
    )
    mnist_test = vision.datasets.FashionMNIST(
        root=root,
        train=False,
        transform=transform,
        download=download,
        source_url=source_url,
    )
    return mnist_train, mnist_test


# reference: http://tangshusen.me/Dive-into-DL-PyTorch/#/chapter03_DL-basics/3.10_mlp-pytorch
def load_data_fashion_mnist(
    batch_size,
    resize=None,
    root="./data-test/fashion-mnist",
    download=True,
    source_url=None,
    num_workers=0,
):
    """Download the Fashion-MNIST dataset and then load into memory."""
    root = os.path.expanduser(root)
    trans = []
    if resize:
        trans.append(transforms.Resize(resize))
    trans.append(transforms.ToTensor())
    transform = transforms.Compose(trans)

    mnist_train = vision.datasets.FashionMNIST(
        root=root,
        train=True,
        transform=transform,
        download=download,
        source_url=source_url,
    )
    mnist_test = vision.datasets.FashionMNIST(
        root=root,
        train=False,
        transform=transform,
        download=download,
        source_url=source_url,
    )

    train_iter = flow.utils.data.DataLoader(
        mnist_train, batch_size, shuffle=True, num_workers=num_workers
    )
    test_iter = flow.utils.data.DataLoader(
        mnist_test, batch_size, shuffle=False, num_workers=num_workers
    )
    return train_iter, test_iter
