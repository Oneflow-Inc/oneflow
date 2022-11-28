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
import numpy as np
import oneflow as flow
from collections import OrderedDict
from oneflow.test_utils.automated_test_util import *
from oneflow.test_utils.test_util import GenArgList

# TODO(): random_tensor can't generate a tensor with nan or inf element.
def _test_isnan(test_case, shape, dtype, device):
    np_array = np.random.randn(*shape)
    mask = np.random.choice([1, 0], np_array.shape, p=[0.1, 0.9]).astype(bool)
    np_array[mask] = np.nan
    of_tensor = flow.tensor(np_array, dtype=dtype, device=device)
    res = flow.isnan(of_tensor)
    test_case.assertTrue(np.allclose(res.numpy(), np.isnan(of_tensor.numpy())))


def _test_isinf(test_case, shape, dtype, device):
    np_array = np.random.randn(*shape)
    mask = np.random.choice([1, 0], np_array.shape, p=[0.1, 0.9]).astype(bool)
    np_array[mask] = np.inf
    of_tensor = flow.tensor(np_array, dtype=dtype, device=device)
    res = flow.isinf(of_tensor)
    test_case.assertTrue(np.allclose(res.numpy(), np.isinf(of_tensor.numpy())))


def _test_isfinite(test_case, shape, dtype, device):
    np_array = np.random.randn(*shape)
    inf_mask = np.random.choice([1, 0], np_array.shape, p=[0.1, 0.9]).astype(bool)
    nan_mask = np.random.choice([1, 0], np_array.shape, p=[0.1, 0.9]).astype(bool)
    np_array[inf_mask] = np.inf
    np_array[nan_mask] = np.nan
    of_tensor = flow.tensor(np_array, dtype=dtype, device=device)
    res = flow.isfinite(of_tensor)
    test_case.assertTrue(np.allclose(res.numpy(), np.isfinite(of_tensor.numpy())))


@flow.unittest.skip_unless_1n1d()
class TestUtilOps(flow.unittest.TestCase):
    def test_util_ops(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [_test_isnan, _test_isinf, _test_isfinite]
        arg_dict["shape"] = [(2, 3, 4), (1, 2, 3)]
        arg_dict["dtype"] = [flow.float, flow.int]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
