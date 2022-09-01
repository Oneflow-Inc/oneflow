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


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
@flow.unittest.skip_unless_1n1d()
class TestSavedTensorHooks(flow.unittest.TestCase):
    def test_unpack_returns_non_tensor(test_case):
        x = flow.ones(1, 2, 3).to("cuda").requires_grad_()
        y = flow.zeros(1, 2, 3).to("cuda").requires_grad_()

        def pack(x):
            return x

        def unpack(x):
            return 0

        with flow.autograd.graph.saved_tensors_hooks(pack, unpack):
            z = x * y
        with test_case.assertRaises(Exception) as exp:
            z.sum().backward()
        test_case.assertTrue(
            "unpack_hook should return a Tensor, but got `<class 'int'>` instead"
            in str(exp.exception)
        )


if __name__ == "__main__":
    unittest.main()
