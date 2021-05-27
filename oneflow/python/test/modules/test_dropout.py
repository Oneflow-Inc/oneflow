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


def _test_dropout(test_case, device):
    input_arr = np.array(
        [
            [-0.7797, 0.2264, 0.2458, 0.4163],
            [0.4299, 0.3626, -0.4892, 0.4141],
            [-1.4115, 1.2183, -0.5503, 0.6520],
        ]
    )
    m = flow.nn.Dropout(p=0)
    x = flow.Tensor(input_arr, device=flow.device(device))
    y = m(x)
    test_case.assertTrue(np.allclose(y.numpy(), input_arr))


def _test_dropout_p1(test_case, device):
    input_arr = np.array(
        [
            [-0.7797, 0.2264, 0.2458, 0.4163],
            [0.4299, 0.3626, -0.4892, 0.4141],
            [-1.4115, 1.2183, -0.5503, 0.6520],
        ]
    )
    output = np.array(
        [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0],]
    )
    m = flow.nn.Dropout(p=1.0)
    x = flow.Tensor(input_arr, device=flow.device(device))
    y = m(x)
    test_case.assertTrue(np.allclose(y.numpy(), output))


def _test_dropout_backward_p0(test_case, device):
    input_arr = np.random.randn(2, 3, 4, 5)
    m = flow.nn.Dropout(p=0)
    x = flow.Tensor(input_arr, device=flow.device(device), requires_grad=True)
    y = m(x)
    z = y.sum()
    z.backward()
    test_case.assertTrue(
        np.allclose(x.grad.numpy(), np.ones((2, 3, 4, 5), dtype=np.float32), 1e-5, 1e-5)
    )


def _test_dropout_backward_p1(test_case, device):
    input_arr = np.random.randn(2, 3, 4, 5)
    m = flow.nn.Dropout(p=1)
    x = flow.Tensor(input_arr, device=flow.device(device), requires_grad=True)
    y = m(x)
    z = y.sum()
    z.backward()
    test_case.assertTrue(
        np.allclose(
            x.grad.numpy(), np.zeros((2, 3, 4, 5), dtype=np.float32), 1e-5, 1e-5
        )
    )


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestDropout(flow.unittest.TestCase):
    def test_transpose(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_functions"] = [
            _test_dropout,
            _test_dropout_p1,
            _test_dropout_backward_p0,
            _test_dropout_backward_p1,
        ]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
