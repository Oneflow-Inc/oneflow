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

import numpy as np
import oneflow as flow
import oneflow.unittest
import unittest


class TestAllReduce(flow.unittest.TestCase):
    # @flow.unittest.skip_unless_1n2d()
    # def test_all_reduce_1n2d(test_case):
    #     np_arr = np.array([[1, 2], [3, 4]])
    #     input = flow.tensor(np_arr, device="cuda")
    #     out = flow.comm.all_reduce(input)
    #     test_case.assertTrue(np.allclose(out.numpy(), np_arr * 2))

    # @flow.unittest.skip_unless_2n2d()
    # def test_all_reduce_2n2d(test_case):
    #     np_arr = np.array([[1, 2], [3, 4]])
    #     input = flow.tensor(np_arr, device="cuda")
    #     out = flow.comm.all_reduce(input)
    #     test_case.assertTrue(np.allclose(out.numpy(), np_arr * 4))

    @flow.unittest.skip_unless_1n2d()
    def test_docs(test_case):
        oneflow.framework.unittest.check_rank0_docstr(oneflow.comm.primitive)

if __name__ == "__main__":
    unittest.main()
