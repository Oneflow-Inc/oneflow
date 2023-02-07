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
from threading import Thread

import numpy as np
import os

import oneflow as flow
import oneflow.unittest


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
class TestComm(flow.unittest.TestCase):
    def _test_send_recv(test_case, x0, src, dst):
        rank = flow.env.get_rank()
        if rank == src:
            x1 = x0
            flow.comm.send(x1, dst)

            x2 = x0
            flow.comm.send(x2, dst)
        elif rank == dst:
            x1 = flow.comm.recv(src)
            test_case.assertTrue(np.array_equal(x1.numpy(), x0.numpy()))
            test_case.assertEqual(x1.device, x0.device)

            x2 = flow.zeros_like(x0)
            flow.comm.recv(src, out=x2)
            test_case.assertTrue(np.array_equal(x2.numpy(), x0.numpy()))
            test_case.assertEqual(x2.device, x0.device)
        else:
            # do nothing
            pass

    @flow.unittest.skip_unless_1n2d()
    def test_send_recv_2_devices(test_case):
        x0 = flow.tensor([[1, 2]])
        test_case._test_send_recv(x0, 0, 1)
        x0 = x0.to("cuda")
        test_case._test_send_recv(x0, 1, 0)

    @flow.unittest.skip_unless_1n4d()
    def test_send_recv_4_devices(test_case):
        x0 = flow.tensor([[1, 2]])
        test_case._test_send_recv(x0, 3, 1)
        x0 = x0.to("cuda")
        test_case._test_send_recv(x0, 0, 3)

    def _test_send_recv_without_sending_meta(test_case, x0, src, dst):
        rank = flow.env.get_rank()
        if rank == src:
            x1 = x0
            flow.comm.send(x1, dst, send_meta=False)

            x2 = x0
            flow.comm.send(x2, dst, send_meta=False)
        elif rank == dst:
            x1 = flow.comm.recv(src, shape=x0.shape, dtype=x0.dtype, device=x0.device)
            test_case.assertTrue(np.array_equal(x1.numpy(), x0.numpy()))

            x2 = flow.zeros_like(x0)
            flow.comm.recv(
                src, shape=x0.shape, dtype=x0.dtype, device=x0.device, out=x2
            )
            test_case.assertTrue(np.array_equal(x2.numpy(), x0.numpy()))
        else:
            # do nothing
            pass

    @flow.unittest.skip_unless_1n2d()
    def test_send_recv_without_sending_meta_2_devices(test_case):
        x0 = flow.tensor([[1, 2]])
        test_case._test_send_recv_without_sending_meta(x0, 1, 0)
        x0 = x0.to("cuda")
        test_case._test_send_recv_without_sending_meta(x0, 0, 1)

    @flow.unittest.skip_unless_1n4d()
    def test_send_recv_without_sending_meta_4_devices(test_case):
        x0 = flow.tensor([[1, 2]])
        test_case._test_send_recv_without_sending_meta(x0, 2, 3)
        x0 = x0.to("cuda")
        test_case._test_send_recv_without_sending_meta(x0, 3, 1)

    @flow.unittest.skip_unless_1n2d()
    def test_comm_in_thread(test_case):
        def threaded_function():
            rank = flow.env.get_rank()
            rev = flow.framework.check_point_v2._broadcast_py_object(rank, 0)
            test_case.assertEqual(rev, 0)

            x = flow.tensor([rank, rank + 1]).to_global(
                placement=flow.placement.all("cpu"), sbp=flow.sbp.split(0)
            )
            test_case.assertTrue(np.array_equal(x.numpy(), np.array([0, 1, 1, 2])))
            x = flow.tensor([rank, rank + 1])
            flow.comm.all_reduce(x)
            test_case.assertTrue(np.array_equal(x.numpy(), np.array([1, 3])))

        thread = Thread(target=threaded_function)
        thread.start()
        thread.join()


if __name__ == "__main__":
    unittest.main()
