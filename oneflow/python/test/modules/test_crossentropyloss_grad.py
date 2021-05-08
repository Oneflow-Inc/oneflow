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
import oneflow as flow
import unittest
import numpy as np


def gen_random_input():
    return np.array(
        [
            [1.1909, -1.5726, 0.9973, -0.7698, -1.1273],
            [1.1354, -1.1815, -1.0553, -0.6178, -2.1103],
        ]
    )


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestCrossEntropyLossModuleGrad(flow.unittest.TestCase):
    def test_CrossEntropyLoss_mean(test_case):
        label = flow.Tensor(np.array([0, 1]), dtype=flow.int32)
        predict = flow.Tensor(np.ones([2, 5]), requires_grad=True)

        CrossEntropyLoss = flow.nn.CrossEntropyLoss(reduction="mean")
        loss = CrossEntropyLoss(predict, label)
        loss.backward()
        target = np.array(
            [
                [-0.4000, 0.1000, 0.1000, 0.1000, 0.1000],
                [0.1000, -0.4000, 0.1000, 0.1000, 0.1000],
            ]
        )

        test_case.assertTrue(predict.grad is not None)
        test_case.assertTrue(
            np.allclose(predict.grad.numpy(), target, rtol=1e-4, atol=1e-8)
        )

    def test_CrossEntropyLoss_sum(test_case):
        label = flow.Tensor(np.array([0, 1]), dtype=flow.int32)
        predict = flow.Tensor(np.ones([2, 5]), requires_grad=True)

        CrossEntropyLoss = flow.nn.CrossEntropyLoss(reduction="sum")
        loss = CrossEntropyLoss(predict, label)
        loss.backward()
        target = np.array(
            [
                [-0.8000, 0.2000, 0.2000, 0.2000, 0.2000],
                [0.2000, -0.8000, 0.2000, 0.2000, 0.2000],
            ]
        )

        test_case.assertTrue(predict.grad is not None)
        test_case.assertTrue(
            np.allclose(predict.grad.numpy(), target, rtol=1e-4, atol=1e-8)
        )

    def test_CrossEntropyLoss_none(test_case):
        label = flow.Tensor(np.array([0, 1]), dtype=flow.int32)
        predict = flow.Tensor(np.ones([2, 5]), requires_grad=True)

        CrossEntropyLoss = flow.nn.CrossEntropyLoss(reduction="none")
        loss = CrossEntropyLoss(predict, label)
        grad = flow.Tensor(np.ones([2]))
        loss.backward(grad)
        target = np.array(
            [
                [-0.8000, 0.2000, 0.2000, 0.2000, 0.2000],
                [0.2000, -0.8000, 0.2000, 0.2000, 0.2000],
            ]
        )

        test_case.assertTrue(predict.grad is not None)
        test_case.assertTrue(
            np.allclose(predict.grad.numpy(), target, rtol=1e-4, atol=1e-8)
        )

    def test_CrossEntropyLoss_mean_with_random_input(test_case):
        label = flow.Tensor(np.array([0, 1]), dtype=flow.int32)
        predict = flow.Tensor(gen_random_input(), requires_grad=True)

        CrossEntropyLoss = flow.nn.CrossEntropyLoss(reduction="mean")
        loss = CrossEntropyLoss(predict, label)
        loss.backward()
        target = np.array(
            [
                [-0.2648, 0.0148, 0.1938, 0.0331, 0.0232],
                [0.3515, -0.4654, 0.0393, 0.0609, 0.0137],
            ]
        )

        test_case.assertTrue(predict.grad is not None)
        test_case.assertTrue(
            np.allclose(predict.grad.numpy(), target, rtol=1e-2, atol=1e-8)
        )

    def test_CrossEntropyLoss_sum_with_random_input(test_case):
        label = flow.Tensor(np.array([0, 1]), dtype=flow.int32)
        predict = flow.Tensor(gen_random_input(), requires_grad=True)

        CrossEntropyLoss = flow.nn.CrossEntropyLoss(reduction="sum")
        loss = CrossEntropyLoss(predict, label)
        loss.backward()
        target = np.array(
            [
                [-0.5297, 0.0297, 0.3875, 0.0662, 0.0463],
                [0.7029, -0.9307, 0.0786, 0.1218, 0.0274],
            ]
        )

        test_case.assertTrue(predict.grad is not None)
        test_case.assertTrue(
            np.allclose(predict.grad.numpy(), target, rtol=1e-2, atol=1e-8)
        )

    def test_CrossEntropyLoss_none_with_random_input(test_case):
        label = flow.Tensor(np.array([0, 1]), dtype=flow.int32)
        predict = flow.Tensor(gen_random_input(), requires_grad=True)

        CrossEntropyLoss = flow.nn.CrossEntropyLoss(reduction="none")
        loss = CrossEntropyLoss(predict, label)
        grad = flow.Tensor(np.ones([2]))
        loss.backward(grad)
        target = np.array(
            [
                [-0.5297, 0.0297, 0.3875, 0.0662, 0.0463],
                [0.7029, -0.9307, 0.0786, 0.1218, 0.0274],
            ]
        )

        test_case.assertTrue(predict.grad is not None)
        test_case.assertTrue(
            np.allclose(predict.grad.numpy(), target, rtol=1e-2, atol=1e-8)
        )


if __name__ == "__main__":
    unittest.main()
