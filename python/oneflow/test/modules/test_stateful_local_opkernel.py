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

import os
import unittest

import oneflow as flow
import oneflow.unittest

import numpy as np


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
class TestStatefulLocalKernel(flow.unittest.TestCase):
    @flow.unittest.skip_unless_1n1d()
    def test_dynamic_attrs(test_case):
        x = (
            flow.builtin_op("constant")
            .Output("out")
            .Attr("is_floating_value", True)
            .Attr("floating_value", 3.0)
            .Attr("dtype", flow.float32)
            .Attr("shape", [2, 3])
            .Build()()[0]
        )
        op = flow.builtin_op("expand_dims").Input("in").Output("out").Build()
        y = op(x, axis=1)[0]
        test_case.assertEqual(y.shape, flow.Size((2, 1, 3)))
        y = op(x, axis=2)[0]
        test_case.assertEqual(y.shape, flow.Size((2, 3, 1)))

    @flow.unittest.skip_unless_1n1d()
    def test_stateful_local_kernel(test_case):
        op1 = (
            flow.builtin_op("constant")
            .Output("out")
            .Attr("is_floating_value", True)
            .Attr("floating_value", 3.0)
            .Attr("dtype", flow.float32)
            .Attr("shape", [1, 1])
            .Build()
        )
        op2 = (
            flow.builtin_op("matmul")
            .Input("a")
            .Input("b")
            .Attr("transpose_a", False)
            .Attr("transpose_b", False)
            .Attr("alpha", float(1.0))
            .Output("out")
            .Build()
        )
        x = op1()[0]
        x = op2(x, x)[0]

    @flow.unittest.skip_unless_1n2d()
    def test_stateful_local_kernel_in_consistent_mode(test_case):
        rank = int(os.getenv("RANK"))

        x = flow.tensor(np.array([1, 2]) * (rank + 1)).to("cuda")
        x = x.to_consistent(flow.placement("cuda", {0: range(2)}), flow.sbp.split(0))

        y = flow.tensor([3, 4, 5]).to("cuda")
        y = y.to_consistent(flow.placement("cuda", {0: range(2)}), flow.sbp.broadcast)

        # logical slice assign op needs sbp and logical shape from stateful local opkernel
        x[:3] = y

        x = x.to_consistent(sbp=flow.sbp.broadcast)

        test_case.assertTrue(
            np.array_equal(x.to_local().numpy(), np.array([3, 4, 5, 4]))
        )


if __name__ == "__main__":
    unittest.main()
