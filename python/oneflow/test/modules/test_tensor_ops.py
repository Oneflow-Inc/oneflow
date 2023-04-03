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
from random import shuffle
from oneflow.test_utils.test_util import GenArgList

import oneflow as flow
import oneflow.unittest

from oneflow.test_utils.automated_test_util import *


def _test_is_floating_point(test_case, shape, device, dtype):
    np_input = np.random.rand(*shape)
    input = flow.tensor(np_input, dtype=dtype, device=device)
    output = input.is_floating_point()
    if input.dtype in (flow.float, flow.float16, flow.float32, flow.double):
        test_case.assertEqual(output, True)
    else:
        test_case.assertEqual(output, False)


def _test_type_dtype(test_case, shape, device, src_dtype, tgt_dtype):
    # test tensor.type(x: dtype) rather than tensor.type_dtype
    np_input = np.random.rand(*shape)
    input = flow.tensor(np_input, dtype=src_dtype, device=device)
    input = input.type(tgt_dtype)
    test_case.assertEqual(input.dtype, tgt_dtype)
    test_case.assertEqual(input.device, flow.device(device))


def _test_type_str(
    test_case, tensortype_dict, shape, device, dtype, tgt_tensortype_str
):
    # test tensor.type(x: str) rather than tensor.type_tensortype
    np_input = np.random.rand(*shape)
    input = flow.tensor(np_input, dtype=dtype, device=device)
    input = input.type(tgt_tensortype_str)
    tgt_dtype, tgt_device = tensortype_dict[tgt_tensortype_str]
    test_case.assertEqual(input.dtype, tgt_dtype)
    test_case.assertEqual(input.device, tgt_device)


def _test_type_tensortype(
    test_case, tensortype_dict, shape, device, dtype, tgt_tensortype
):
    # test tensor.type(x: tensortype) rather than tensor.type_tensortype
    np_input = np.random.rand(*shape)
    input = flow.tensor(np_input, dtype=dtype, device=device)
    input = input.type(tgt_tensortype)
    tgt_dtype, tgt_device = tensortype_dict[tgt_tensortype]
    test_case.assertEqual(input.dtype, tgt_dtype)
    test_case.assertEqual(input.device, tgt_device)


def _test_type_noargs(test_case, shape, device, dtype):
    # test tensor.type() rather than tensor.type_noargs
    def generate_tensortype_string(device, dtype):
        dtype_to_str_dict = {
            flow.uint8: "ByteTensor",
            flow.int8: "CharTensor",
            flow.int32: "IntTensor",
            flow.int64: "LongTensor",
            flow.float16: "HalfTensor",
            flow.bfloat16: "BFloat16Tensor",  # Currently unsupport
            flow.float32: "FloatTensor",
            flow.float64: "DoubleTensor",
        }
        dtype = dtype_to_str_dict[dtype]
        if device == "cpu":
            return dtype
        return ".".join([device, dtype])

    np_input = np.random.rand(*shape)
    input = flow.tensor(np_input, dtype=dtype, device=device)
    test_case.assertEqual(
        input.type(), "oneflow." + generate_tensortype_string(device, dtype)
    )


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

    @autotest(n=5, auto_backward=False)
    def test_long_with_non_contiguous_input(test_case):
        device = random_device()
        permute_list = list(range(4))
        shuffle(permute_list)
        input = random_tensor(ndim=4).to(device)
        x = input.permute(permute_list)
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
    def test_half(test_case):
        device = random_device()
        x = random_tensor(dtype=int).to(device)
        y = x.half()
        return y

    @autotest(n=20, auto_backward=False, rtol=1e-4, atol=1e-4, check_graph=True)
    def test_half_0dim(test_case):
        device = random_device()
        x = random_tensor(ndim=0, dtype=int).to(device)
        y = x.half()
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

    @autotest(n=20, auto_backward=False, rtol=1e-4, atol=1e-4, check_graph=True)
    def test_bool(test_case):
        device = random_device()
        x = random_tensor().to(device)
        y = x.bool()
        return y

    @autotest(n=20, auto_backward=False, rtol=1e-4, atol=1e-4, check_graph=True)
    def test_bool_0dim(test_case):
        device = random_device()
        x = random_tensor(ndim=0).to(device)
        y = x.bool()
        return y

    @autotest(n=5, auto_backward=False)
    def test_bool_with_non_contiguous_input(test_case):
        device = random_device()
        permute_list = list(range(4))
        shuffle(permute_list)
        input = random_tensor(ndim=4).to(device)
        x = input.permute(permute_list)
        y = x.bool()
        return y

    # Not check graph because of 2 reason.
    # Reason 1, nn.Graph.build()'s input/output item only support types: Tensor/None.
    # Reason 2, This op needs to convert the EagerTensor to a numpy array，so this op only supports eager mode.
    # Please refer to File "oneflow/api/python/utils/tensor_utils.h", line 49, in EagerTensorToNumpy.
    @autotest(
        n=20, auto_backward=False, rtol=1e-4, atol=1e-4, check_graph="ValidatedFalse"
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
        n=20, auto_backward=False, rtol=1e-4, atol=1e-4, check_graph="ValidatedFalse"
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
        n=20, auto_backward=False, rtol=1e-4, atol=1e-4, check_graph="ValidatedFalse"
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
        n=20, auto_backward=False, rtol=1e-4, atol=1e-4, check_graph="ValidatedFalse"
    )
    def test_tolist_0dim(test_case):
        device = random_device()
        x = random_tensor(ndim=0).to(device)
        y = torch.tensor(x.tolist())
        return y

    @autotest()
    def test_type_as(test_case):
        input = random_tensor().to(random_device())
        target = random_tensor().to(random_device())
        input = input.type_as(target)
        return input

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

    def test_type_dtype(test_case):
        # test tensor.type(x.dtype) rather than tensor.type_dtype
        arg_dict = OrderedDict()
        arg_dict["shape"] = [(1, 2), (3, 4, 5), (2, 3, 4, 5)]
        arg_dict["device"] = ["cpu", "cuda"]
        arg_dict["src_dtype"] = [
            flow.uint8,
            flow.int8,
            flow.int64,
            flow.int32,
            flow.float16,
            flow.float32,
            flow.float64,
        ]
        arg_dict["tgt_dtype"] = arg_dict["src_dtype"]
        for arg in GenArgList(arg_dict):
            _test_type_dtype(test_case, *arg)

    def test_type_tensortype_str_cpu(test_case):
        # test tensor.type(x: str) rather than tensor.type_tensortype
        arg_dict = OrderedDict()
        arg_dict["shape"] = [(1, 2), (3, 4, 5), (2, 3, 4, 5)]
        arg_dict["device"] = ["cpu", "cuda"]
        arg_dict["src_dtype"] = [
            flow.uint8,
            flow.int8,
            flow.int64,
            flow.int32,
            flow.float16,
            flow.float32,
            flow.float64,
        ]
        tensortype_dict = {
            "oneflow.CharTensor": [flow.int8, flow.device("cpu")],
            "oneflow.ByteTensor": [flow.uint8, flow.device("cpu")],
            "oneflow.IntTensor": [flow.int32, flow.device("cpu")],
            "oneflow.LongTensor": [flow.int64, flow.device("cpu")],
            "oneflow.HalfTensor": [flow.float16, flow.device("cpu")],
            "oneflow.FloatTensor": [flow.float32, flow.device("cpu")],
            "oneflow.DoubleTensor": [flow.float64, flow.device("cpu")],
        }
        arg_dict["tgt_tensortype_str"] = list(tensortype_dict.keys())
        for arg in GenArgList(arg_dict):
            _test_type_str(test_case, tensortype_dict, *arg)

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_type_tensortype_str(test_case):
        # test tensor.type(x: str) rather than tensor.type_tensortype
        arg_dict = OrderedDict()
        arg_dict["shape"] = [(1, 2), (3, 4, 5), (2, 3, 4, 5)]
        arg_dict["device"] = ["cpu", "cuda"]
        arg_dict["src_dtype"] = [
            flow.uint8,
            flow.int8,
            flow.int64,
            flow.int32,
            flow.float16,
            flow.float32,
            flow.float64,
        ]
        tensortype_dict = {
            "oneflow.CharTensor": [flow.int8, flow.device("cpu")],
            "oneflow.ByteTensor": [flow.uint8, flow.device("cpu")],
            "oneflow.IntTensor": [flow.int32, flow.device("cpu")],
            "oneflow.LongTensor": [flow.int64, flow.device("cpu")],
            "oneflow.HalfTensor": [flow.float16, flow.device("cpu")],
            "oneflow.FloatTensor": [flow.float32, flow.device("cpu")],
            "oneflow.DoubleTensor": [flow.float64, flow.device("cpu")],
            "oneflow.cuda.CharTensor": [flow.int8, flow.device("cuda")],
            "oneflow.cuda.ByteTensor": [flow.uint8, flow.device("cuda")],
            "oneflow.cuda.IntTensor": [flow.int32, flow.device("cuda")],
            "oneflow.cuda.LongTensor": [flow.int64, flow.device("cuda")],
            "oneflow.cuda.HalfTensor": [flow.float16, flow.device("cuda")],
            "oneflow.cuda.FloatTensor": [flow.float32, flow.device("cuda")],
            "oneflow.cuda.DoubleTensor": [flow.float64, flow.device("cuda")],
        }
        arg_dict["tgt_tensortype_str"] = list(tensortype_dict.keys())
        for arg in GenArgList(arg_dict):
            _test_type_str(test_case, tensortype_dict, *arg)

    def test_type_tensortype_cpu(test_case):
        # test tensor.type(x: tensortype) rather than tensor.type_tensortype
        arg_dict = OrderedDict()
        arg_dict["shape"] = [(1, 2), (3, 4, 5), (2, 3, 4, 5)]
        arg_dict["device"] = ["cpu", "cuda"]
        arg_dict["src_dtype"] = [
            flow.uint8,
            flow.int8,
            flow.int64,
            flow.int32,
            flow.float16,
            flow.float32,
            flow.float64,
        ]
        tensortype_dict = {
            flow.CharTensor: [flow.int8, flow.device("cpu")],
            flow.ByteTensor: [flow.uint8, flow.device("cpu")],
            flow.IntTensor: [flow.int32, flow.device("cpu")],
            flow.LongTensor: [flow.int64, flow.device("cpu")],
            flow.HalfTensor: [flow.float16, flow.device("cpu")],
            flow.FloatTensor: [flow.float32, flow.device("cpu")],
            flow.DoubleTensor: [flow.float64, flow.device("cpu")],
        }
        arg_dict["tgt_tensortype"] = list(tensortype_dict.keys())
        for arg in GenArgList(arg_dict):
            _test_type_tensortype(test_case, tensortype_dict, *arg)

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_type_tensortype(test_case):
        # test tensor.type(x: tensortype) rather than tensor.type_tensortype
        arg_dict = OrderedDict()
        arg_dict["shape"] = [(1, 2), (3, 4, 5), (2, 3, 4, 5)]
        arg_dict["device"] = ["cpu", "cuda"]
        arg_dict["src_dtype"] = [
            flow.uint8,
            flow.int8,
            flow.int64,
            flow.int32,
            flow.float16,
            flow.float32,
            flow.float64,
        ]
        tensortype_dict = {
            flow.CharTensor: [flow.int8, flow.device("cpu")],
            flow.ByteTensor: [flow.uint8, flow.device("cpu")],
            flow.IntTensor: [flow.int32, flow.device("cpu")],
            flow.LongTensor: [flow.int64, flow.device("cpu")],
            flow.HalfTensor: [flow.float16, flow.device("cpu")],
            flow.Tensor: [flow.float32, flow.device("cpu")],
            flow.FloatTensor: [flow.float32, flow.device("cpu")],
            flow.DoubleTensor: [flow.float64, flow.device("cpu")],
            flow.cuda.CharTensor: [flow.int8, flow.device("cuda")],
            flow.cuda.ByteTensor: [flow.uint8, flow.device("cuda")],
            flow.cuda.IntTensor: [flow.int32, flow.device("cuda")],
            flow.cuda.LongTensor: [flow.int64, flow.device("cuda")],
            flow.cuda.HalfTensor: [flow.float16, flow.device("cuda")],
            flow.cuda.FloatTensor: [flow.float32, flow.device("cuda"),],
            flow.cuda.DoubleTensor: [flow.float64, flow.device("cuda"),],
        }
        arg_dict["tgt_tensortype"] = list(tensortype_dict.keys())
        for arg in GenArgList(arg_dict):
            _test_type_tensortype(test_case, tensortype_dict, *arg)

    def test_type_noargs(test_case):
        # test tensor.type() rather than tensor.type_noargs
        arg_dict = OrderedDict()
        arg_dict["shape"] = [(1, 2), (3, 4, 5), (2, 3, 4, 5)]
        arg_dict["device"] = ["cpu", "cuda"]
        arg_dict["dtype"] = [
            flow.uint8,
            flow.int8,
            flow.int64,
            flow.int32,
            flow.float16,
            flow.float32,
            flow.float64,
        ]
        for arg in GenArgList(arg_dict):
            _test_type_noargs(test_case, *arg)

    @autotest(n=3, auto_backward=False)
    def test_bincount(test_case):
        device = random_device()
        len = random(1, 100)
        input = random_tensor(1, len, dtype=int, low=0).to(device)
        weight = random_tensor(1, len, dtype=float).to(device)
        min_length = random(1, 100) | nothing()
        return (
            input.bincount(minlength=min_length),
            input.bincount(weight, minlength=min_length),
        )


if __name__ == "__main__":
    unittest.main()
