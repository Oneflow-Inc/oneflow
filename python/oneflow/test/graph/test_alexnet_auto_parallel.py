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
import argparse

import numpy as np

import oneflow as flow
import oneflow.unittest
from alexnet_model import alexnet
import flowvision as vision
import flowvision.transforms as transforms


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


def _parse_args():
    parser = argparse.ArgumentParser("flags for train alexnet")
    parser.add_argument(
        "--load_checkpoint", type=str, default="", help="load checkpoint"
    )
    parser.add_argument(
        "--ofrecord_path",
        type=str,
        default="/dataset/imagenette/ofrecord",
        help="dataset path",
    )
    # training hyper-parameters
    parser.add_argument(
        "--learning_rate", type=float, default=0.02, help="learning rate"
    )
    parser.add_argument("--mom", type=float, default=0.9, help="momentum")
    parser.add_argument("--epochs", type=int, default=1, help="training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="val batch size")

    return parser.parse_known_args()


def _test_alexnet_graph(test_case, args, placement, sbp):
    data_dir = os.path.join(
        os.getenv("ONEFLOW_TEST_CACHE_DIR", "./data-test"), "fashion-mnist-lenet"
    )
    source_url = "https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/mnist/Fashion-MNIST/"
    train_iter, test_iter = load_data_fashion_mnist(
        batch_size=args.batch_size,
        root=data_dir,
        download=True,
        source_url=source_url,
        num_workers=0,
        resize=(112, 112),
    )

    # oneflow init
    start_t = time.time()
    alexnet_module = alexnet(num_classes=10)
    end_t = time.time()
    print("init time : {}".format(end_t - start_t))

    alexnet_module.to_global(placement, sbp)

    of_cross_entropy = flow.nn.CrossEntropyLoss().to_global(placement, sbp)

    of_sgd = flow.optim.SGD(
        alexnet_module.parameters(), lr=args.learning_rate, momentum=args.mom
    )

    class AlexNetGraph(flow.nn.Graph):
        def __init__(self):
            super().__init__()
            self.alexnet = alexnet_module
            self.cross_entropy = of_cross_entropy
            self.add_optimizer(of_sgd)
            self.config.enable_auto_parallel(True)
            self.config.enable_auto_parallel_ignore_user_sbp_config(True)
            self.config.enable_auto_parallel_trunk_algo(True)
            self.config.enable_auto_parallel_sbp_collector(True)

        def build(self, image, label):
            logits = self.alexnet(image)
            loss = self.cross_entropy(logits, label)
            loss.backward()
            return loss

    alexnet_graph = AlexNetGraph()

    class AlexNetEvalGraph(flow.nn.Graph):
        def __init__(self):
            super().__init__()
            self.alexnet = alexnet_module
            self.config.enable_auto_parallel(True)
            self.config.enable_auto_parallel_ignore_user_sbp_config(True)
            self.config.enable_auto_parallel_trunk_algo(True)
            self.config.enable_auto_parallel_sbp_collector(True)

        def build(self, image):
            with flow.no_grad():
                logits = self.alexnet(image)
                predictions = logits.softmax()
            return predictions

    alexnet_eval_graph = AlexNetEvalGraph()

    of_losses = []
    print_interval = 20

    acc = 0.0
    for epoch in range(args.epochs):
        alexnet_module.train()

        for i, (image, label) in enumerate(train_iter):
            # oneflow graph train
            if image.shape[0] != args.batch_size:
                # drop last batch
                break
            start_t = time.time()
            image = image.to_global(placement, sbp).expand(args.batch_size, 3, 112, 112)
            label = label.to_global(placement, sbp)
            loss = alexnet_graph(image, label)
            end_t = time.time()
            if i % print_interval == 0:
                l = loss.numpy()
                of_losses.append(l)
                if flow.env.get_rank() == 0:
                    print(
                        "epoch {} train iter {}/{} oneflow loss {}, train time : {}".format(
                            epoch, i, len(train_iter), l, end_t - start_t
                        )
                    )
        if flow.env.get_rank() == 0:
            print("epoch %d train done, start validation" % epoch)

        alexnet_module.eval()
        correct_of = 0.0
        total_of = 0.0
        for image, label in test_iter:
            # oneflow graph eval
            if image.shape[0] != args.batch_size:
                # drop last batch
                break
            start_t = time.time()
            image = image.to_global(placement, sbp).expand(args.batch_size, 3, 112, 112)
            predictions = alexnet_eval_graph(image)
            of_predictions = predictions.numpy()
            clsidxs = np.argmax(of_predictions, axis=1)

            label_nd = label.numpy()

            for i in range(args.batch_size):
                total_of += 1
                if clsidxs[i] == label_nd[i]:
                    correct_of += 1
            end_t = time.time()
        acc = correct_of / total_of

        if flow.env.get_rank() == 0:
            print("epoch %d, oneflow top1 val acc: %f" % (epoch, acc))
    #  test_case.assertTrue(acc > 0.50)


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
class TestAlexnetAutoParallel(oneflow.unittest.TestCase):
    def test_alexnet_auto_parallel_1d_sbp(test_case):
        args, unknown_args = _parse_args()
        placement = flow.placement.all("cuda")
        sbp = [flow.sbp.broadcast,] * len(placement.ranks.shape)
        _test_alexnet_graph(test_case, args, placement, sbp)


if __name__ == "__main__":
    unittest.main()
