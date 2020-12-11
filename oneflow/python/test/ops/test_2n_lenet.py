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
# lenet_train.py
import oneflow as flow
import oneflow.typing as tp
import unittest

BATCH_SIZE = 100


def lenet(data, train=False):
    initializer = flow.truncated_normal(0.1)
    conv1 = flow.layers.conv2d(
        data,
        32,
        5,
        padding="SAME",
        activation=flow.nn.relu,
        kernel_initializer=initializer,
        name="conv1",
    )
    pool1 = flow.nn.max_pool2d(conv1, ksize=2, strides=2, padding="SAME", name="pool1")
    conv2 = flow.layers.conv2d(
        pool1,
        64,
        5,
        padding="SAME",
        activation=flow.nn.relu,
        kernel_initializer=initializer,
        name="conv2",
    )
    pool2 = flow.nn.max_pool2d(conv2, ksize=2, strides=2, padding="SAME", name="pool2")
    reshape = flow.reshape(pool2, [pool2.shape[0], -1])
    # with flow.scope.placement("gpu", ["1:0"]):
    with flow.scope.placement("gpu", ["0:0-1"]):
        # with flow.scope.placement("gpu", ["0:0-1","1:0-1"]):
        hidden = flow.layers.dense(
            reshape,
            512,
            activation=flow.nn.relu,
            kernel_initializer=initializer,
            name="hidden",
            model_distribute=flow.distribute.split(axis=0),  ###
        )
    if train:
        hidden = flow.nn.dropout(hidden, rate=0.5)

    output = flow.layers.dense(
        hidden,
        10,
        kernel_initializer=initializer,
        name="outlayer",
        model_distribute=flow.distribute.split(axis=0),
    )
    return output


@flow.unittest.skip_unless_2n2d()
class Test2dGpuVariable(flow.unittest.TestCase):
    def test_2n_lenet(test_case):
        @flow.global_function(type="train")
        def train_job(
            images: tp.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
            labels: tp.Numpy.Placeholder((BATCH_SIZE,), dtype=flow.int32),
        ) -> tp.Numpy:
            with flow.scope.placement("gpu", "0:0"):
                logits = lenet(images, train=True)
                loss = flow.nn.sparse_softmax_cross_entropy_with_logits(
                    labels, logits, name="softmax_loss"
                )

            lr_scheduler = flow.optimizer.PiecewiseConstantScheduler([], [0.1])
            flow.optimizer.SGD(lr_scheduler, momentum=0).minimize(loss)
            return loss

        flow.config.gpu_device_num(2)
        check_point = flow.train.CheckPoint()
        check_point.init()

        (train_images, train_labels), (test_images, test_labels) = flow.data.load_mnist(
            BATCH_SIZE, BATCH_SIZE
        )

        for epoch in range(20):
            for i, (images, labels) in enumerate(zip(train_images, train_labels)):
                loss = train_job(images, labels)
                if i % 20 == 0:
                    print(loss.mean())
        check_point.save("./lenet_models_1")  # need remove the existed folder
        print("model saved")


if __name__ == "__main__":
    unittest.main()
