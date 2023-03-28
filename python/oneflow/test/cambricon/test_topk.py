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


def _test_topk(test_case, shape, dtype):
    x_np = np.random.randn(*shape)
    dim = random.choice(list(range(x_np.ndim)) + [None])
    k = np.random.randint(1, x_np.shape[dim] if dim is not None else x_np.shape[-1])

    def _get_result(device):
        x = flow.tensor(x_np, dtype=dtype, device=flow.device(device))
        values, indices = flow.topk(x, k=k, dim=dim)
        return values.numpy(), indices.numpy()

    values_cpu, indices_cpu = _get_result("cpu")
    values_mlu, indices_mlu = _get_result("mlu")
    test_case.assertTrue(np.allclose(values_cpu, values_mlu, 0.0001, 0.0001))
    test_case.assertTrue(np.allclose(indices_cpu, indices_mlu, 0.0001, 0.0001))


@flow.unittest.skip_unless_1n1d()
class TestTopKCambriconModule(flow.unittest.TestCase):
    def test_topk(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_topk,
        ]
        arg_dict["shape"] = [
            (16, 32,),
            (8, 12, 24),
        ]
        arg_dict["dtype"] = [
            flow.float32,
            flow.uint8,
            flow.int8,
            flow.int32,
        ]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
