import os
import oneflow as flow



def load_data_cifar10(
    batch_size,
    data_dir="./data-test/cifar10",
    download=True,
    transform=None,
    source_url=None,
    num_workers=0,
):
    cifar10_train = flow.utils.vision.datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=download,
        transform=transform,
        source_url=source_url,
    )
    cifar10_test = flow.utils.vision.datasets.CIFAR10(
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


def get_fashion_mnist_dataset(
    resize=None,
    root="./data-test/fashion-mnist",
    download=True,
    source_url=None,
):
    root = os.path.expanduser(root)
    trans = []
    if resize:
        trans.append(flow.utils.vision.transforms.Resize(resize))
    trans.append(flow.utils.vision.transforms.ToTensor())
    transform = flow.utils.vision.transforms.Compose(trans)

    mnist_train = flow.utils.vision.datasets.FashionMNIST(
        root=root,
        train=True,
        transform=transform,
        download=download,
        source_url=source_url,
    )
    mnist_test = flow.utils.vision.datasets.FashionMNIST(
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
        trans.append(flow.utils.vision.transforms.Resize(resize))
    trans.append(flow.utils.vision.transforms.ToTensor())
    transform = flow.utils.vision.transforms.Compose(trans)

    mnist_train = flow.utils.vision.datasets.FashionMNIST(
        root=root,
        train=True,
        transform=transform,
        download=download,
        source_url=source_url,
    )
    mnist_test = flow.utils.vision.datasets.FashionMNIST(
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
