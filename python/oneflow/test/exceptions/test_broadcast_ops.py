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

import oneflow as flow
import oneflow.unittest

binary_ops = [
    flow.add,
    flow.sub,
    flow.mul,
    flow.div,
    flow.min,
    flow.minimum,
    flow.max,
    flow.maximum,
    flow.fmod,
    flow.pow,
    flow.eq,
    flow.ne,
    flow.gt,
    flow.ge,
    flow.lt,
    flow.le,
    flow.logical_and,
    flow.logical_or,
    flow.logical_xor,
]


@flow.unittest.skip_unless_1n1d()
class TestBroadcastOps(flow.unittest.TestCase):
    def test_broadcast_binary_ops(test_case):
        x = flow.Tensor(8, 10)
        y = flow.Tensor(8)
        for op in binary_ops:
            with test_case.assertRaises(RuntimeError) as ctx:
                op(x, y)
            test_case.assertTrue(
                "The size of tensor a (10) must match the size of tensor b (8) at non-singleton dimension 1"
                in str(ctx.exception)
            )


if __name__ == "__main__":
    unittest.main()
