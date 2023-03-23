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

def _test_unsorted_segment_sum_like(test_case, shape, indes, device, dtype):
    table = flow.tensor(np.random.randn(*shape), device=flow.device(device), dtype=dtype)
    indes = flow.tensor(indes, device=flow.device(device), dtype=flow.int32)
    grad_shape = (*indes.shape, *table.shape[1:])
    out_grad = flow.tensor(np.random.randn(*grad_shape), device=flow.device(device), dtype=dtype)
    mlu_out = flow._C.unsorted_segment_sum_like(out_grad, indes, table, axis=0)
    tol = 1e-8
    if dtype == flow.float16:
        out_grad = out_grad.to(flow.float32)
        table = table.to(flow.float32)
        tol = 0.001
    cpu_out = flow._C.unsorted_segment_sum_like(out_grad.cpu(), indes.cpu(), table.cpu(), axis=0)
    test_case.assertTrue(
        np.allclose(mlu_out.to(flow.float32).numpy(), cpu_out.numpy(), atol=tol, rtol=tol)
    )


@flow.unittest.skip_unless_1n1d()
class TestUnsortedSegmentSumLikeCambriconModule(flow.unittest.TestCase):
    def test_unsorted_segment_sum_like(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_unsorted_segment_sum_like,
        ]
        arg_dict["shape"] = [(10, 3),]
        arg_dict["indes"] = [[[1, 2, 4, 5], [4, 3, 2, 9]],]
        arg_dict["device"] = ["mlu"]
        arg_dict["dtype"] = [
            flow.float32,
            flow.float16,
        ]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])

if __name__ == "__main__":
    unittest.main()
