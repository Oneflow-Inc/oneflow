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
import unittest
from collections import OrderedDict

import numpy as np

import oneflow.experimental as flow
from test_util import GenArgList


def _gen_random_input():
    return np.array(
        [
            [1.1909, -1.5726, 0.9973, -0.7698, -1.1273],
            [1.1354, -1.1815, -1.0553, -0.6178, -2.1103],
        ]
    )


def _test_CrossEntropyLoss_mean(test_case, device):
    label = flow.Tensor(np.array([0, 1]), dtype=flow.int32, device=flow.device(device))
    predict = flow.Tensor(
        np.ones([2, 5]), requires_grad=True, device=flow.device(device)
    )

    CrossEntropyLoss = flow.nn.CrossEntropyLoss(reduction="mean")
    CrossEntropyLoss = CrossEntropyLoss.to(device)
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


def _test_CrossEntropyLoss_sum(test_case, device):
    label = flow.Tensor(np.array([0, 1]), dtype=flow.int32, device=flow.device(device))
    predict = flow.Tensor(
        np.ones([2, 5]), requires_grad=True, device=flow.device(device)
    )

    CrossEntropyLoss = flow.nn.CrossEntropyLoss(reduction="sum")
    CrossEntropyLoss = CrossEntropyLoss.to(device)
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


def _test_CrossEntropyLoss_none(test_case, device):
    label = flow.Tensor(np.array([0, 1]), dtype=flow.int32, device=flow.device(device))
    predict = flow.Tensor(
        np.ones([2, 5]), requires_grad=True, device=flow.device(device)
    )

    CrossEntropyLoss = flow.nn.CrossEntropyLoss(reduction="none")
    CrossEntropyLoss = CrossEntropyLoss.to(device)
    loss = CrossEntropyLoss(predict, label)
    loss = loss.sum()
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


def _test_CrossEntropyLoss_mean_with_random_input(test_case, device):
    label = flow.Tensor(np.array([0, 1]), dtype=flow.int32, device=flow.device(device))
    predict = flow.Tensor(
        _gen_random_input(), requires_grad=True, device=flow.device(device)
    )

    CrossEntropyLoss = flow.nn.CrossEntropyLoss(reduction="mean")
    CrossEntropyLoss = CrossEntropyLoss.to(device)
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


def _test_CrossEntropyLoss_sum_with_random_input(test_case, device):
    label = flow.Tensor(np.array([0, 1]), dtype=flow.int32, device=flow.device(device))
    predict = flow.Tensor(
        _gen_random_input(), requires_grad=True, device=flow.device(device)
    )

    CrossEntropyLoss = flow.nn.CrossEntropyLoss(reduction="sum")
    CrossEntropyLoss = CrossEntropyLoss.to(device)
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


def _test_CrossEntropyLoss_none_with_random_input(test_case, device):
    label = flow.Tensor(np.array([0, 1]), dtype=flow.int32, device=flow.device(device))
    predict = flow.Tensor(
        _gen_random_input(), requires_grad=True, device=flow.device(device)
    )

    CrossEntropyLoss = flow.nn.CrossEntropyLoss(reduction="none")
    CrossEntropyLoss = CrossEntropyLoss.to(device)
    loss = CrossEntropyLoss(predict, label)
    loss = loss.sum()
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


def _test_CrossEntropyLoss_none_with_ignore_index(test_case, device):
    label = flow.Tensor(np.array([0, 1]), dtype=flow.int32, device=flow.device(device))
    predict = flow.Tensor(
        np.ones([2, 5]), requires_grad=True, device=flow.device(device)
    )

    CrossEntropyLoss = flow.nn.CrossEntropyLoss(reduction="none", ignore_index=1)
    CrossEntropyLoss = CrossEntropyLoss.to(device)
    loss = CrossEntropyLoss(predict, label)
    loss = loss.sum()
    loss.backward()
    target = np.array(
        [[-0.8000, 0.2000, 0.2000, 0.2000, 0.2000], [0.0, 0.0, 0.0, 0.0, 0.0],]
    )

    test_case.assertTrue(predict.grad is not None)
    test_case.assertTrue(
        np.allclose(predict.grad.numpy(), target, rtol=1e-4, atol=1e-8)
    )


def _test_CrossEntropyLoss_mean_with_random_input_with_ignore_index(test_case, device):
    label = flow.Tensor(np.array([0, 1]), dtype=flow.int32, device=flow.device(device))
    predict = flow.Tensor(
        _gen_random_input(), requires_grad=True, device=flow.device(device)
    )

    CrossEntropyLoss = flow.nn.CrossEntropyLoss(reduction="mean", ignore_index=0)
    CrossEntropyLoss = CrossEntropyLoss.to(device)
    loss = CrossEntropyLoss(predict, label)
    loss.backward()
    target = np.array(
        [[0.0, 0.0, 0.0, 0.0, 0.0], [0.7030, -0.9307, 0.0786, 0.1218, 0.0274],]
    )

    test_case.assertTrue(predict.grad is not None)
    test_case.assertTrue(
        np.allclose(predict.grad.numpy(), target, rtol=1e-2, atol=1e-8)
    )


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestCrossEntropyLossModuleGrad(flow.unittest.TestCase):
    def test_crossentropyloss_grad(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_CrossEntropyLoss_mean,
            _test_CrossEntropyLoss_sum,
            _test_CrossEntropyLoss_none,
            _test_CrossEntropyLoss_mean_with_random_input,
            _test_CrossEntropyLoss_sum_with_random_input,
            _test_CrossEntropyLoss_none_with_random_input,
            _test_CrossEntropyLoss_none_with_ignore_index,
            _test_CrossEntropyLoss_mean_with_random_input_with_ignore_index,
        ]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
