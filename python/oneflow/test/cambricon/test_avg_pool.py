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
import random
from collections import OrderedDict

import numpy as np
from oneflow.test_utils.test_util import GenArgList

import oneflow as flow
import oneflow.unittest


def _test_avg_pool2d_forward(
    test_case,
    shape,
    kernel,
    stride,
    padding,
    count_include_pad,
    ceil_mode,
    device,
    dtype,
):
    x = flow.tensor(np.random.randn(*shape), device=flow.device(device), dtype=dtype)
    kwargs = {
        "kernel_size": kernel,
        "stride": stride,
        "padding": padding,
        "count_include_pad": count_include_pad,
        "ceil_mode": ceil_mode,
    }
    mlu_result = flow.nn.functional.avg_pool2d(x, **kwargs)
    cpu_result = flow.nn.functional.avg_pool2d(x.cpu().float(), **kwargs)
    test_case.assertTrue(
        np.allclose(mlu_result.cpu().numpy(), cpu_result.numpy(), 0.0001, 0.0001)
    )


def _test_avg_pool2d_backward(
    test_case,
    shape,
    kernel,
    stride,
    padding,
    count_include_pad,
    ceil_mode,
    device,
    dtype,
):
    x = np.random.randn(*shape)
    mlu_x = flow.tensor(x, device=flow.device(device), dtype=dtype, requires_grad=True)
    cpu_x = flow.tensor(x, dtype=dtype, requires_grad=True)
    kwargs = {
        "kernel_size": kernel,
        "stride": stride,
        "padding": padding,
        "count_include_pad": count_include_pad,
        "ceil_mode": ceil_mode,
    }
    mlu_result = flow.nn.functional.avg_pool2d(mlu_x, **kwargs)
    cpu_result = flow.nn.functional.avg_pool2d(cpu_x, **kwargs)
    test_case.assertTrue(
        np.allclose(mlu_result.cpu().numpy(), cpu_result.numpy(), 0.0001, 0.0001)
    )
    mlu_result.sum().backward()
    cpu_result.sum().backward()
    test_case.assertTrue(
        np.allclose(mlu_x.grad.cpu().numpy(), cpu_x.grad.numpy(), 0.0001, 0.0001)
    )


@flow.unittest.skip_unless_1n1d()
class TestMaxPoolCambriconModule(flow.unittest.TestCase):
    def test_avg_pool2d(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_avg_pool2d_forward,
            _test_avg_pool2d_backward,
        ]
        arg_dict["shape"] = [
            (1, 3, 24, 24),
            (3, 3, 112, 112),
        ]
        arg_dict["kernel"] = [
            2,
            3,
            [2, 3],
        ]
        arg_dict["stride"] = [
            None,
            2,
            3,
            [2, 3],
        ]
        arg_dict["padding"] = [
            0,
            1,
            [0, 1],
        ]
        arg_dict["count_include_pad"] = [True, False]
        arg_dict["ceil_mode"] = [True, False]
        arg_dict["device"] = ["mlu"]
        arg_dict["dtype"] = [
            flow.float32,
        ]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
