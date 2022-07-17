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
import argparse
import numpy as np
import os
import time
import unittest

import oneflow as flow
import oneflow.unittest

from alexnet_model import alexnet
from ofrecord_data_utils import OFRecordDataLoader


def _parse_args():
    parser = argparse.ArgumentParser("flags for train alexnet")
    parser.add_argument(
        "--save_checkpoint_path",
        type=str,
        default="./checkpoints",
        help="save checkpoint root dir",
    )
    parser.add_argument(
        "--load_checkpoint", type=str, default="", help="load checkpoint"
    )
    parser.add_argument(
        "--ofrecord_path",
        type=str,
        default="/dataset/imagenette/ofrecord",
        help="dataset path",
    )
    parser.add_argument(
        "--train_dataset_size", type=int, default=400, help="train_dataset size"
    )
    parser.add_argument(
        "--val_dataset_size", type=int, default=40, help="val_dataset size"
    )
    # training hyper-parameters
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="learning rate"
    )
    parser.add_argument("--mom", type=float, default=0.9, help="momentum")
    parser.add_argument("--epochs", type=int, default=1, help="training epochs")
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="train batch size"
    )
    parser.add_argument("--val_batch_size", type=int, default=4, help="val batch size")
    parser.add_argument("--device", type=str, default="cuda", help="device")

    return parser.parse_known_args()


def _test_alexnet_graph_repr(test_case, args):
    train_data_loader = OFRecordDataLoader(
        ofrecord_root=args.ofrecord_path,
        mode="train",
        dataset_size=args.train_dataset_size,
        batch_size=args.train_batch_size,
    )

    alexnet_module = alexnet()
    alexnet_module.to(args.device)

    of_cross_entropy = flow.nn.CrossEntropyLoss()
    of_cross_entropy.to(args.device)

    of_sgd = flow.optim.SGD(
        alexnet_module.parameters(), lr=args.learning_rate, momentum=args.mom
    )

    class AlexNetGraph(flow.nn.Graph):
        def __init__(self):
            super().__init__()
            self.alexnet = alexnet_module
            self.cross_entropy = of_cross_entropy
            self.add_optimizer(of_sgd)

        def build(self, image, label):
            logits = self.alexnet(image)
            loss = self.cross_entropy(logits, label)
            loss.backward()
            return loss

    alexnet_graph = AlexNetGraph()

    print("repr(alexnet_graph) before run: \n", repr(alexnet_graph))

    # debug graph build
    alexnet_graph.debug(1, op_repr_with_py_stack=True, max_py_stack_depth=4)

    alexnet_module.train()
    image, label = train_data_loader()
    image = image.to(args.device)
    label = label.to(args.device)
    loss = alexnet_graph(image, label)

    print("repr(alexnet_graph) after run: \n", repr(alexnet_graph))


def _test_alexnet_graph(test_case, args):
    train_data_loader = OFRecordDataLoader(
        ofrecord_root=args.ofrecord_path,
        mode="train",
        dataset_size=args.train_dataset_size,
        batch_size=args.train_batch_size,
    )
    val_data_loader = OFRecordDataLoader(
        ofrecord_root=args.ofrecord_path,
        mode="val",
        dataset_size=args.val_dataset_size,
        batch_size=args.val_batch_size,
    )

    # oneflow init
    start_t = time.time()
    alexnet_module = alexnet()
    end_t = time.time()
    print("init time : {}".format(end_t - start_t))

    alexnet_module.to(args.device)

    of_cross_entropy = flow.nn.CrossEntropyLoss()
    of_cross_entropy.to(args.device)

    of_sgd = flow.optim.SGD(
        alexnet_module.parameters(), lr=args.learning_rate, momentum=args.mom
    )

    class AlexNetGraph(flow.nn.Graph):
        def __init__(self):
            super().__init__()
            self.train_data_loader = train_data_loader
            self.alexnet = alexnet_module
            self.cross_entropy = of_cross_entropy
            self.add_optimizer(of_sgd)

        def build(self):
            image, label = self.train_data_loader()
            image = image.to(args.device)
            label = label.to(args.device)
            logits = self.alexnet(image)
            loss = self.cross_entropy(logits, label)
            loss.backward()
            return loss

    alexnet_graph = AlexNetGraph()

    class AlexNetEvalGraph(flow.nn.Graph):
        def __init__(self):
            super().__init__()
            self.val_data_loader = val_data_loader
            self.alexnet = alexnet_module

        def build(self):
            with flow.no_grad():
                image, label = self.val_data_loader()
                image = image.to(args.device)
                logits = self.alexnet(image)
                predictions = logits.softmax()
            return predictions, label

    alexnet_eval_graph = AlexNetEvalGraph()

    of_losses = []
    all_samples = len(val_data_loader) * args.val_batch_size
    print_interval = 10

    for epoch in range(args.epochs):
        alexnet_module.train()

        for b in range(len(train_data_loader)):
            # oneflow graph train
            start_t = time.time()
            loss = alexnet_graph()
            end_t = time.time()
            if b % print_interval == 0:
                l = loss.numpy()
                of_losses.append(l)
                print(
                    "epoch {} train iter {} oneflow loss {}, train time : {}".format(
                        epoch, b, l, end_t - start_t
                    )
                )
        print("epoch %d train done, start validation" % epoch)

        alexnet_module.eval()
        correct_of = 0.0
        for b in range(len(val_data_loader)):

            start_t = time.time()
            predictions, label = alexnet_eval_graph()
            of_predictions = predictions.numpy()
            clsidxs = np.argmax(of_predictions, axis=1)

            label_nd = label.numpy()
            for i in range(args.val_batch_size):
                if clsidxs[i] == label_nd[i]:
                    correct_of += 1
            end_t = time.time()

        print("epoch %d, oneflow top1 val acc: %f" % (epoch, correct_of / all_samples))


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
@flow.unittest.skip_unless_1n1d()
class TestAlexnetGraph(oneflow.unittest.TestCase):
    def test_alexnet_graph_repr(test_case):
        args, unknown_args = _parse_args()
        args.device = "cuda"
        _test_alexnet_graph_repr(test_case, args)

    def test_alexnet_graph_gpu(test_case):
        args, unknown_args = _parse_args()
        args.device = "cuda"
        _test_alexnet_graph(test_case, args)

    def test_alexnet_graph_cpu(test_case):
        args, unknown_args = _parse_args()
        args.device = "cpu"
        args.train_batch_size = 40
        _test_alexnet_graph(test_case, args)


if __name__ == "__main__":
    unittest.main()
