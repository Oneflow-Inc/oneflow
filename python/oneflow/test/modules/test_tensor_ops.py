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


def _test_type_as(test_case, shape, device, src_dtype, tgt_dtype):
    np_input = np.random.rand(*shape)
    input = flow.tensor(np_input, dtype=src_dtype, device=device)
    target = flow.tensor(np_input, dtype=tgt_dtype, device=device)
    input = input.type_as(target)
    test_case.assertEqual(input.dtype, target.dtype)


def _test_is_floating_point(test_case, shape, device, dtype):
    np_input = np.random.rand(*shape)
    input = flow.tensor(np_input, dtype=dtype, device=device)
    output = input.is_floating_point()
    if input.dtype in (flow.float, flow.float16, flow.float32, flow.double):
        test_case.assertEqual(output, True)
    else:
        test_case.assertEqual(output, False)


@flow.unittest.skip_unless_1n1d()
@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
class TestCuda(flow.unittest.TestCase):
    @autotest(n=20, auto_backward=True, rtol=1e-4, atol=1e-4, check_graph=True)
    def test_cuda(test_case):
        device = random_device()
        x = random_tensor().to(device)
        x = x.cuda()
        y = x.sum()
        return y

    @autotest(n=20, auto_backward=True, rtol=1e-4, atol=1e-4, check_graph=True)
    def test_cuda_0dim(test_case):
        device = random_device()
        x = random_tensor(ndim=0).to(device)
        x = x.cuda()
        y = x.sum()
        return y

    @autotest(n=5)
    def test_cuda_int_device(test_case):
        device = random_device()
        x = random_tensor().to(device)
        x = x.cuda(0)
        y = x.sum()
        return y


@flow.unittest.skip_unless_1n1d()
class TestTensorOps(flow.unittest.TestCase):
    @autotest(n=20, auto_backward=False, rtol=1e-4, atol=1e-4, check_graph=True)
    def test_cpu(test_case):
        device = random_device()
        x = random_tensor().to(device)
        x = x.cpu()
        y = x.sum()
        return y

    @autotest(n=20, auto_backward=False, rtol=1e-4, atol=1e-4, check_graph=True)
    def test_long(test_case):
        device = random_device()
        x = random_tensor().to(device)
        y = x.long()
        return y

    @autotest(n=20, auto_backward=False, rtol=1e-4, atol=1e-4, check_graph=True)
    def test_long_0dim(test_case):
        device = random_device()
        x = random_tensor(ndim=0).to(device)
        y = x.long()
        return y

    @autotest(n=20, auto_backward=False, rtol=1e-4, atol=1e-4, check_graph=True)
    def test_int(test_case):
        device = random_device()
        x = random_tensor().to(device)
        y = x.int()
        return y

    @autotest(n=20, auto_backward=False, rtol=1e-4, atol=1e-4, check_graph=True)
    def test_int_0dim(test_case):
        device = random_device()
        x = random_tensor(ndim=0).to(device)
        y = x.int()
        return y

    @autotest(n=20, auto_backward=False, rtol=1e-4, atol=1e-4, check_graph=True)
    def test_float(test_case):
        device = random_device()
        x = random_tensor(dtype=int).to(device)
        y = x.float()
        return y

    @autotest(n=20, auto_backward=False, rtol=1e-4, atol=1e-4, check_graph=True)
    def test_float_0dim(test_case):
        device = random_device()
        x = random_tensor(ndim=0, dtype=int).to(device)
        y = x.float()
        return y

    @autotest(n=20, auto_backward=False, rtol=1e-4, atol=1e-4, check_graph=True)
    def test_double(test_case):
        device = random_device()
        x = random_tensor(dtype=int).to(device)
        y = x.double()
        return y

    @autotest(n=20, auto_backward=False, rtol=1e-4, atol=1e-4, check_graph=True)
    def test_double_0dim(test_case):
        device = random_device()
        x = random_tensor(ndim=0, dtype=int).to(device)
        y = x.double()
        return y

    # Not check graph because of 2 reason.
    # Reason 1, nn.Graph.build()'s input/output item only support types: Tensor/None.
    # Reason 2, This op needs to convert the EagerTensor to a numpy array，so this op only supports eager mode.
    # Please refer to File "oneflow/api/python/utils/tensor_utils.h", line 49, in EagerTensorToNumpy.
    @autotest(
        n=20, auto_backward=False, rtol=1e-4, atol=1e-4, check_graph="ValidatedFlase"
    )
    def test_item(test_case):
        device = random_device()
        x = random_tensor(ndim=1, dim0=1, dtype=int).to(device)
        y = torch.tensor(x.item())
        return y

    # Not check graph because of 2 reason.
    # Reason 1, nn.Graph.build()'s input/output item only support types: Tensor/None.
    # Reason 2, This op needs to convert the EagerTensor to a numpy array，so this op only supports eager mode.
    # Please refer to File "oneflow/api/python/utils/tensor_utils.h", line 49, in EagerTensorToNumpy.
    @autotest(
        n=20, auto_backward=False, rtol=1e-4, atol=1e-4, check_graph="ValidatedFlase"
    )
    def test_item_0dim(test_case):
        device = random_device()
        x = random_tensor(ndim=0, dtype=int).to(device)
        y = torch.tensor(x.item())
        return y

    # Not check graph because of 2 reason.
    # Reason 1, nn.Graph.build()'s input/output item only support types: Tensor/None.
    # Reason 2, This op needs to convert the EagerTensor to a numpy array，so this op only supports eager mode.
    # Please refer to File "oneflow/api/python/utils/tensor_utils.h", line 49, in EagerTensorToNumpy.
    @autotest(
        n=20, auto_backward=False, rtol=1e-4, atol=1e-4, check_graph="ValidatedFlase"
    )
    def test_tolist(test_case):
        device = random_device()
        x = random_tensor(ndim=4).to(device)
        y = torch.tensor(x.tolist())
        return y

    # Not check graph because of 2 reason.
    # Reason 1, nn.Graph.build()'s input/output item only support types: Tensor/None.
    # Reason 2, This op needs to convert the EagerTensor to a numpy array，so this op only supports eager mode.
    # Please refer to File "oneflow/api/python/utils/tensor_utils.h", line 49, in EagerTensorToNumpy.
    @autotest(
        n=20, auto_backward=False, rtol=1e-4, atol=1e-4, check_graph="ValidatedFlase"
    )
    def test_tolist_0dim(test_case):
        device = random_device()
        x = random_tensor(ndim=0).to(device)
        y = torch.tensor(x.tolist())
        return y

    def test_type_as(test_case):
        arg_dict = OrderedDict()
        arg_dict["shape"] = [(1, 2), (3, 4, 5), (2, 3, 4, 5)]
        arg_dict["device"] = ["cpu", "cuda"]
        arg_dict["src_dtype"] = [flow.int64, flow.int32, flow.float32, flow.float64]
        arg_dict["tgt_dtype"] = [flow.int64, flow.int32, flow.float32, flow.float64]
        for arg in GenArgList(arg_dict):
            _test_type_as(test_case, *arg)

    def test_is_floating_point(test_case):
        arg_dict = OrderedDict()
        arg_dict["shape"] = [(1, 2), (3, 4, 5), (2, 3, 4, 5)]
        arg_dict["device"] = ["cpu", "cuda"]
        arg_dict["dtype"] = [
            flow.uint8,
            flow.int8,
            flow.int32,
            flow.int64,
            flow.float32,
            flow.float64,
            flow.double,
            flow.float,
            flow.int,
        ]
        for arg in GenArgList(arg_dict):
            _test_is_floating_point(test_case, *arg)


if __name__ == "__main__":
    unittest.main()
