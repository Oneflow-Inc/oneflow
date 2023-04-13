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
from oneflow.test_utils.test_util import GenArgList

import oneflow as flow
import oneflow.unittest

from oneflow.test_utils.automated_test_util import *


def np_nonzero(input, as_tuple):
    if as_tuple:
        return np.nonzero(input)
    else:
        return np.transpose(np.nonzero(input))


def _test_nonzero(test_case, shape, as_tuple, device):
    np_input = np.random.randn(*shape)
    input = flow.tensor(np_input, dtype=flow.float32, device=flow.device(device))
    of_out = flow.nonzero(input, as_tuple)
    np_out = np_nonzero(np_input, as_tuple)
    if as_tuple:
        test_case.assertTrue(
            np.allclose(tuple(x.numpy() for x in of_out), np_out, 0.0001, 0.0001)
        )
    else:
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 0.0001, 0.0001))


@flow.unittest.skip_unless_1n1d()
class TestNonzero(flow.unittest.TestCase):
    def test_nonzero(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [_test_nonzero]
        arg_dict["shape"] = [(2, 3), (2, 3, 4), (2, 4, 5, 6), (2, 3, 0, 4)]
        arg_dict["as_tuple"] = [True, False]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])

    # Not check graph because of one reason:
    # Reason 1, lazy tensor cannot call .numpy(). tensor.numpy() is not allowed to called in nn.Graph.build(*args) or called by lazy tensor.
    # Please refer to File "python/oneflow/nn/modules/nonzero.py", line 29, in nonzero_op.
    @autotest(auto_backward=False, check_graph="ValidatedFalse")
    def test_nonzero_with_random_data(test_case):
        device = random_device()
        x = random_tensor(ndim=random(2, 5).to(int)).to(device)
        y = torch.nonzero(x)
        return y

    # Not check graph because of one reason:
    # Reason 1, lazy tensor cannot call .numpy(). tensor.numpy() is not allowed to called in nn.Graph.build(*args) or called by lazy tensor.
    # Please refer to File "python/oneflow/nn/modules/nonzero.py", line 29, in nonzero_op.
    @autotest(auto_backward=False, check_graph="ValidatedFalse")
    def test_nonzero_bool_with_random_data(test_case):
        device = random_device()
        x = random_tensor(ndim=random(2, 5).to(int)).to(device=device, dtype=torch.bool)
        y = torch.nonzero(x)
        return y

    # Not check graph because of one reason:
    # Reason 1, lazy tensor cannot call .numpy(). tensor.numpy() is not allowed to called in nn.Graph.build(*args) or called by lazy tensor.
    @autotest(auto_backward=False, check_graph="ValidatedFalse")
    def test_half_nonzero_with_random_data(test_case):
        device = random_device()
        x = random_tensor(ndim=random(2, 5).to(int)).to(
            device=device, dtype=torch.float16
        )
        y = torch.nonzero(x)
        return y

    # Not check graph because of one reason:
    # Reason 1, lazy tensor cannot call .numpy(). tensor.numpy() is not allowed to called in nn.Graph.build(*args) or called by lazy tensor.
    # Please refer to File "python/oneflow/nn/modules/nonzero.py", line 29, in nonzero_op.
    @autotest(auto_backward=False, check_graph="ValidatedFalse")
    def test_nonzero_with_0dim_data(test_case):
        device = random_device()
        x = random_tensor(ndim=0).to(device)
        y = torch.nonzero(x)
        return y

    # Not check graph because of one reason:
    # Reason 1, lazy tensor cannot call .numpy(). tensor.numpy() is not allowed to called in nn.Graph.build(*args) or called by lazy tensor.
    # Please refer to File "python/oneflow/nn/modules/nonzero.py", line 29, in nonzero_op.
    @autotest(auto_backward=False, check_graph="ValidatedFalse")
    def test_nonzero_tuple_with_random_data(test_case):
        device = random_device()
        x = random_tensor(ndim=random(2, 5).to(int)).to(device)
        y = torch.nonzero(x, as_tuple=True)
        return y


if __name__ == "__main__":
    unittest.main()
