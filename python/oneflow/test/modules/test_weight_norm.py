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
import oneflow as flow
import oneflow.unittest

from oneflow.test_utils.test_util import GenArgList

import torch as torch_original
from oneflow.test_utils.automated_test_util import *

input_arr = np.array(
    [
        [-0.16046895, -1.03667831],
        [-0.34974465, 0.26505867],
        [-1.24111986, -0.53806001],
        [1.72426331, 0.43572459],
    ],
    dtype=np.float64,
)


def _test_weightnorm(test_case, device, dim):
    model_flow = flow.nn.Linear(2, 4)
    model_flow = model_flow.to(device)
    with flow.no_grad():
        for i in range(input_arr.shape[0]):
            for j in range(input_arr.shape[1]):
                model_flow.weight[i, j] = input_arr[i][j]
    m_flow = flow.nn.utils.weight_norm(model_flow, name="weight", dim=dim)

    model_torch = torch_original.nn.Linear(2, 4)
    model_torch = model_torch.to(device)
    with torch_original.no_grad():
        for i in range(input_arr.shape[0]):
            for j in range(input_arr.shape[1]):
                model_torch.weight[i, j] = input_arr[i][j]
    m_torch = torch_original.nn.utils.weight_norm(model_torch, name="weight", dim=dim)

    if device == "cpu":
        test_case.assertTrue(
            np.allclose(
                m_flow.weight_g.detach().numpy(),
                m_torch.weight_g.detach().numpy(),
                1e-05,
                1e-05,
            )
        )
        test_case.assertTrue(
            np.allclose(
                m_flow.weight_v.detach().numpy(),
                m_torch.weight_v.detach().numpy(),
                1e-05,
                1e-05,
            )
        )
    elif device == "cuda":
        test_case.assertTrue(
            np.allclose(
                m_flow.weight_g.detach().cpu().numpy(),
                m_torch.weight_g.detach().cpu().numpy(),
                1e-05,
                1e-05,
            )
        )
        test_case.assertTrue(
            np.allclose(
                m_flow.weight_v.detach().numpy(),
                m_torch.weight_v.detach().cpu().numpy(),
                1e-05,
                1e-05,
            )
        )


def _test_weightnorm_backward(test_case, device, dim):
    linear = flow.nn.Linear(3, 8)
    x = flow.tensor(
        [
            [-0.94630778, -0.83378579, -0.87060891],
            [2.0289922, -0.28708987, -2.18369248],
            [0.35217619, -0.67095644, -1.58943879],
            [0.08086036, -1.81075924, 1.20752494],
            [0.8901075, -0.49976737, -1.07153746],
            [-0.44872912, -1.07275683, 0.06256855],
            [-0.22556897, 0.74798368, 0.90416439],
            [0.48339456, -2.32742195, -0.59321527],
        ],
        dtype=flow.float32,
        requires_grad=True,
    )
    flow.nn.init.constant_(linear.weight, 2.068758)
    flow.nn.init.constant_(linear.bias, 0.23)

    linear_wn = flow.nn.utils.weight_norm(linear, name="weight", dim=dim)
    of_out = linear_wn(x)

    of_out = of_out.sum()
    of_out.backward()

    np_grad = np.array(
        [
            [16.5501, 16.5501, 16.5501],
            [16.5501, 16.5501, 16.5501],
            [16.5501, 16.5501, 16.5501],
            [16.5501, 16.5501, 16.5501],
            [16.5501, 16.5501, 16.5501],
            [16.5501, 16.5501, 16.5501],
            [16.5501, 16.5501, 16.5501],
            [16.5501, 16.5501, 16.5501],
        ]
    )
    test_case.assertTrue(np.allclose(np_grad, x.grad.numpy(), 0.0001, 0.0001))


@flow.unittest.skip_unless_1n1d()
class TestWeightNorm(flow.unittest.TestCase):
    def test_weightnorm(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_weightnorm,
            _test_weightnorm_backward,
        ]
        arg_dict["device"] = ["cpu", "cuda"]
        arg_dict["dim"] = [None, -2, -1, 0, 1]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])

    # Not check graph because of one reason:
    # Reason 1, Graph's build input nn.modules.linear.Linear type is not supported.
    # Please refer to issue: https://github.com/Oneflow-Inc/oneflow/issues/7466
    @autotest(n=10, auto_backward=True, check_graph="ValidatedFalse")
    def test_weight_norm_with_random_data(test_case):
        device = random_device()

        dim = random(-2, 2).to(int).value()
        output = random(2, 6).to(int)
        input = random(2, 6).to(int)

        model_torch = torch.nn.Linear(output, input)
        model_torch = model_torch.to(device)
        m = torch.nn.utils.weight_norm(model_torch, name="weight", dim=dim)
        return m.weight_g, m.weight_v


if __name__ == "__main__":
    unittest.main()
