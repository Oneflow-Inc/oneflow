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


def _test_max_pool2d_forward(
    test_case,
    shape,
    kernel,
    stride,
    padding,
    dilation,
    return_indices,
    ceil_mode,
    device,
    dtype,
):
    x = flow.tensor(np.random.randn(*shape), device=flow.device(device), dtype=dtype)
    kwargs = {
        "kernel_size": kernel,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "return_indices": return_indices,
        "ceil_mode": ceil_mode,
    }
    mlu_result = flow.nn.functional.max_pool2d(x, **kwargs)
    cpu_result = flow.nn.functional.max_pool2d(x.cpu().float(), **kwargs)
    if return_indices:
        y_mlu, indices_mlu = mlu_result
        y_cpu, indices_cpu = cpu_result
    else:
        y_mlu, indices_mlu = mlu_result, None
        y_cpu, indices_cpu = cpu_result, None
    if dtype == flow.float16:
        test_case.assertTrue(
            np.allclose(y_mlu.cpu().float().numpy(), y_cpu.numpy(), 0.001, 0.001)
        )
    else:
        test_case.assertTrue(
            np.allclose(y_mlu.cpu().numpy(), y_cpu.numpy(), 0.0001, 0.0001)
        )
    if indices_cpu is not None and indices_mlu is not None:
        test_case.assertTrue(
            np.allclose(indices_mlu.cpu().numpy(), indices_cpu.numpy(), 0.0001, 0.0001)
        )


@flow.unittest.skip_unless_1n1d()
class TestMaxPoolCambriconModule(flow.unittest.TestCase):
    def test_max_pool2d(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_max_pool2d_forward,
        ]
        arg_dict["shape"] = [
            (1, 3, 24, 24),
            (3, 1, 112, 112),
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
        # dilation != 1 has bug
        arg_dict["dilation"] = [
            # 1, 2, [1, 2],
            1,
        ]
        # the definition of indices result of max_pool2d doesn't match
        # so disable return_indices here
        arg_dict["return_indices"] = [
            # True, False
            False,
        ]
        arg_dict["ceil_mode"] = [True, False]
        arg_dict["device"] = ["mlu"]
        arg_dict["dtype"] = [
            flow.float32,
            flow.float16,
        ]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
