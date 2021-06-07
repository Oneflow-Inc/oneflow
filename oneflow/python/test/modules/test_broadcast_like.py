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


def _test_broadcast_like(test_case, device):
    input = flow.Tensor(
        np.ones(shape=(3, 1, 1), dtype=np.float32),
        dtype=flow.float32,
        device=flow.device(device),
    )
    like_tensor = flow.Tensor(
        np.ones(shape=(3, 3, 3), dtype=np.float32),
        dtype=flow.float32,
        device=flow.device(device),
    )
    of_out = flow.broadcast_like(input, like_tensor, broadcast_axes=(1, 2))
    np_out = np.ones(shape=(3, 3, 3))
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-5, 1e-5))


def _test_broadcast_like_3dim(test_case, device):
    input = flow.Tensor(
        np.ones(shape=(1, 3, 2), dtype=np.float32),
        dtype=flow.float32,
        device=flow.device(device),
    )
    like_tensor = flow.Tensor(
        np.ones(shape=(3, 3, 2), dtype=np.float32),
        dtype=flow.float32,
        device=flow.device(device),
    )
    of_out = flow.broadcast_like(input, like_tensor, broadcast_axes=(0,))
    np_out = np.ones(shape=(3, 3, 2))
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-5, 1e-5))


def _test_broadcast_like_4dim(test_case, device):
    input = flow.Tensor(
        np.ones(shape=(1, 3, 2, 1), dtype=np.float32),
        dtype=flow.float32,
        device=flow.device(device),
    )
    like_tensor = flow.Tensor(
        np.ones(shape=(3, 3, 2, 3), dtype=np.float32),
        dtype=flow.float32,
        device=flow.device(device),
    )
    of_out = flow.broadcast_like(input, like_tensor, broadcast_axes=(0, 3))
    np_out = np.ones(shape=(3, 3, 2, 3))
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-5, 1e-5))


def _test_broadcast_like_backward(test_case, device):
    input = flow.Tensor(
        np.ones(shape=(3, 1, 1), dtype=np.float32),
        dtype=flow.float32,
        device=flow.device(device),
        requires_grad=True,
    )
    like_tensor = flow.Tensor(
        np.ones(shape=(3, 3, 3), dtype=np.float32),
        dtype=flow.float32,
        device=flow.device(device),
        requires_grad=True,
    )
    of_out = flow.broadcast_like(input, like_tensor, broadcast_axes=(1, 2))
    of_out = of_out.sum()
    of_out.backward()
    np_grad = [[[9.0]], [[9.0]], [[9.0]]]
    test_case.assertTrue(np.allclose(input.grad.numpy(), np_grad, 1e-5, 1e-5))


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestBroadCastLike(flow.unittest.TestCase):
    def test_broadcast_like(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_broadcast_like,
            _test_broadcast_like_3dim,
            _test_broadcast_like_4dim,
            _test_broadcast_like_backward,
        ]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
