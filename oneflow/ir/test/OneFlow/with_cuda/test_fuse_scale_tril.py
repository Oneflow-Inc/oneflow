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
# RUN: python3 -m oneflow.test_utils.throttle --with-cuda=%with_cuda python3 %s | FileCheck %s
# CHECK-NOT: oneflow.tril

import os
import unittest
import numpy as np

os.environ["ONEFLOW_MLIR_ENABLE_ROUND_TRIP"] = "1"
os.environ["ONEFLOW_MLIR_FUSE_FORWARD_OPS"] = "1"
os.environ["ONEFLOW_MLIR_STDOUT"] = "1"
import oneflow as flow
from collections import OrderedDict
from oneflow.test_utils.test_util import GenArgDict


def _test_fused_scale_tril(
    test_case, shape, diagonal=0, scale=1.0,
):
    x = np.random.rand(*shape)
    # Different dtype will result in insert of cast op causing pass to fail.
    tensor_x = flow.tensor(x, device="cuda", dtype=flow.float32)
    eager_out = flow.tril(tensor_x, diagonal) * scale

    class TestFuseScaleTril(flow.nn.Graph):
        def __init__(self):
            super().__init__()

        def build(self):
            return flow.tril(tensor_x * scale, diagonal)

    lazy_out_0 = TestFuseScaleTril()()
    test_case.assertTrue(np.allclose(eager_out.numpy(), lazy_out_0.numpy()))

    class TestFuseTrilScale(flow.nn.Graph):
        def __init__(self):
            super().__init__()

        def build(self):
            return flow.tril(tensor_x, diagonal) * scale

    lazy_out_1 = TestFuseTrilScale()()
    test_case.assertTrue(np.allclose(eager_out.numpy(), lazy_out_1.numpy()))


@flow.unittest.skip_unless_1n1d()
class FusedScaleTrilTestCase(flow.unittest.TestCase):
    def test_fused_scale_tril(test_case):
        arg_dict = OrderedDict()
        arg_dict["shape"] = [(5, 5), (4, 6)]
        arg_dict["diagonal"] = [-1, 0]
        arg_dict["scale"] = [-2.3, 2.0]
        for kwargs in GenArgDict(arg_dict):
            _test_fused_scale_tril(test_case, **kwargs)


if __name__ == "__main__":
    unittest.main()
