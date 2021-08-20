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
import oneflow.unittest
from automated_test_util import *


class TestComm(flow.unittest.TestCase):
    def _test_send_recv(test_case, x0):
        rank = flow.framework.distribute.get_rank()
        if rank == 0:
            x1 = x0
            flow.comm.send(x1, 1)

            x2 = x0
            flow.comm.send(x2, 1)
        elif rank == 1:
            x1 = flow.comm.recv(0)
            test_case.assertTrue(np.array_equal(x1.numpy(), x0.numpy()))
            test_case.assertEqual(x1.device, x0.device)

            x2 = flow.zeros_like(x0)
            flow.comm.recv(0, out=x2)
            test_case.assertTrue(np.array_equal(x2.numpy(), x0.numpy()))
            test_case.assertEqual(x2.device, x0.device)
        else:
            raise ValueError()

    @flow.unittest.skip_unless_1n2d()
    def test_send_recv(test_case):
        x0 = flow.tensor([[1, 2]])
        test_case._test_send_recv(x0)
        x0 = x0.to("cuda")
        test_case._test_send_recv(x0)

    # @flow.unittest.skip_unless_1n2d()
    # def test_send_recv_with_meta(test_case):
    #     rank = flow.framework.distribute.get_rank()
    #     x0 = flow.tensor([[1, 2]])
    #     if rank == 0:
    #         x1 = x0
    #         flow.F.send_with_meta(x1, 1)
    #
    #         x2 = x0
    #         flow.F.send_with_meta(x2, 1)
    #     elif rank == 1:
    #         x1 = flow.F.recv_with_meta(0, x0.shape, x0.dtype, x0.device)
    #         test_case.assertTrue(np.array_equal(x1.numpy(), x0.numpy()))
    #
    #         x2 = flow.tensor([[0, 0]])
    #         flow.F.recv_with_meta(0, x0.shape, x0.dtype, x0.device, out=x2)
    #         test_case.assertTrue(np.array_equal(x2.numpy(), x0.numpy()))
    #     else:
    #         raise ValueError()
    #

if __name__ == "__main__":
    unittest.main()
