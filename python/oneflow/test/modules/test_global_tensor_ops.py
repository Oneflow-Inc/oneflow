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


def _test_type_as(test_case, shape, src_dtype, tgt_dtype, placement, sbp):
    np_input = np.random.rand(*shape)
    input = flow.tensor(np_input, dtype=src_dtype).to_global(placement, sbp)
    target = flow.tensor(np_input, dtype=tgt_dtype).to_global(placement, sbp)
    input = input.type_as(target)
    test_case.assertEqual(input.dtype, target.dtype)


def _test_local_to_global_type_as(
    test_case, shape, src_dtype, tgt_dtype, placement, sbp
):
    np_input = np.random.rand(*shape)
    input = random_tensor(ndim=len(shape)).oneflow.to_local()
    target = flow.tensor(np_input, dtype=tgt_dtype).to_global(placement, sbp)
    input = input.type_as(target)
    test_case.assertEqual(input.dtype, target.dtype)
    test_case.assertEqual(input.placement, target.placement)
    test_case.assertEqual(input.sbp, target.sbp)


def _test_global_to_local_type_as(
    test_case, shape, src_dtype, tgt_dtype, placement, sbp
):
    np_input = np.random.rand(*shape)
    input = flow.tensor(np_input, dtype=tgt_dtype).to_global(placement, sbp)
    target = random_tensor(ndim=len(shape)).to(random_device()).oneflow.to_local()
    input = input.type_as(target)
    test_case.assertEqual(input.dtype, target.dtype)
    test_case.assertEqual(input.device, target.device)


def _test_is_floating_point(test_case, shape, dtype, placement, sbp):
    np_input = np.random.rand(*shape)
    input = flow.tensor(np_input, dtype=dtype).to_global(placement, sbp)
    output = input.is_floating_point()
    if input.dtype in (flow.float, flow.float16, flow.float32, flow.double):
        test_case.assertEqual(output, True)
    else:
        test_case.assertEqual(output, False)


@autotest(n=1, check_graph=True)
@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
def _test_global_cuda(test_case, placement, sbp):
    x = random_tensor(2, 8, 16).to_global(placement, sbp)
    x = x.cuda()
    y = x.sum()
    return y


class TestGlobalCuda(flow.unittest.TestCase):
    @globaltest
    def test_global_cuda(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=2):
                _test_global_cuda(test_case, placement, sbp)


@autotest(n=1, check_graph=True)
def _test_global_cpu(test_case, placement, sbp):
    x = random_tensor(2, 8, 16).to_global(placement, sbp)
    x = x.cpu()
    y = x.sum()
    return y


# PyTorch error if open auto_backward:
# element 0 of tensors does not require grad and does not have a grad_fn
@autotest(n=1, auto_backward=False, check_graph=True)
def _test_global_long(test_case, placement, sbp):
    x = random_tensor(2, 8, 16, requires_grad=True).to_global(placement, sbp)
    y = x.long()
    test_case.assertFalse(y.oneflow.requires_grad)
    return y


@autotest(n=1, auto_backward=False, check_graph=True)
def _test_global_int(test_case, placement, sbp):
    x = random_tensor(2, 8, 16, requires_grad=True).to_global(placement, sbp)
    y = x.int()
    test_case.assertFalse(y.oneflow.requires_grad)
    return y


@autotest(n=1, auto_backward=False, check_graph=True)
def _test_global_float(test_case, placement, sbp):
    x = random_tensor(2, 8, 16, dtype=int).to_global(placement, sbp)
    y = x.float()
    return y


@autotest(n=1, auto_backward=False, check_graph=True)
def _test_global_double(test_case, placement, sbp):
    x = random_tensor(2, 8, 16, dtype=int).to_global(placement, sbp)
    y = x.double()
    return y


@autotest(n=1, auto_backward=False, check_graph=True)
def _test_global_item(test_case, placement, sbp):
    x = random_tensor(ndim=1, dim0=1, dtype=int).to_global(placement, sbp)
    y = torch.tensor(x.item())
    return y


@autotest(n=1, auto_backward=False, check_graph=False)
def _test_global_tolist(test_case, placement, sbp):
    x = random_tensor(ndim=4, dim0=8, dim1=16, dim2=24, dim3=32, dtype=int).to_global(
        placement, sbp
    )
    y = torch.tensor(x.tolist())
    return y


class TestGlobalTensorOps(flow.unittest.TestCase):
    @globaltest
    def test_global_cpu(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=2):
                _test_global_cpu(test_case, placement, sbp)

    @globaltest
    def test_global_long(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=2):
                _test_global_long(test_case, placement, sbp)

    @globaltest
    def test_global_int(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=2):
                _test_global_int(test_case, placement, sbp)

    @globaltest
    def test_global_float(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=2):
                _test_global_float(test_case, placement, sbp)

    @globaltest
    def test_global_double(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=2):
                _test_global_double(test_case, placement, sbp)

    @unittest.skip("TODO: sometimes global item will result to segment fault!")
    @globaltest
    def test_global_item(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=1, except_split=True):
                _test_global_item(test_case, placement, sbp)

    @globaltest
    def test_global_tolist(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=4):
                _test_global_tolist(test_case, placement, sbp)

    @globaltest
    def test_type_as(test_case):
        arg_dict = OrderedDict()
        arg_dict["shape"] = [(8, 16), (8, 16, 24), (8, 16, 24, 32)]
        arg_dict["src_dtype"] = [flow.int64, flow.int32, flow.float32, flow.float64]
        arg_dict["tgt_dtype"] = [flow.int64, flow.int32, flow.float32, flow.float64]
        for arg in GenArgList(arg_dict):
            for placement in all_placement():
                for sbp in all_sbp(placement, max_dim=len(arg[0])):
                    _test_type_as(test_case, *arg, placement, sbp)
                    _test_local_to_global_type_as(test_case, *arg, placement, sbp)
                    _test_global_to_local_type_as(test_case, *arg, placement, sbp)

    @globaltest
    def test_is_floating_point(test_case):
        arg_dict = OrderedDict()
        arg_dict["shape"] = [(8, 16), (8, 16, 24), (8, 16, 24, 32)]
        arg_dict["dtype"] = [
            # flow.uint8, nccl don't support uint8
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
            for placement in all_placement():
                for sbp in all_sbp(placement, max_dim=len(arg[0])):
                    _test_is_floating_point(test_case, *arg, placement, sbp)


if __name__ == "__main__":
    unittest.main()
