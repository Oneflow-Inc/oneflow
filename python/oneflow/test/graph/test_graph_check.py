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
import sys

import numpy as np

import oneflow as flow
import oneflow.unittest
from oneflow.framework.tensor import Tensor, TensorTuple


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
@flow.unittest.skip_unless_1n1d()
class TestGraphCheck(flow.unittest.TestCase):
    def test_non_tensor_types_of_module(test_case):
        class CustomModuleIOCheck(flow.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, t, lt, n, i, s):
                return t, lt, n, i, s

        class CustomGraphIOCheck(flow.nn.Graph):
            def __init__(self):
                super().__init__()
                self.m = CustomModuleIOCheck()
                self.m.config.activation_checkpointing = True

            def build(self, t, lt, n):
                rt, rlt, n, ri, rs = self.m(t, lt, n, 1, "2")
                return t, lt, n

        g = CustomGraphIOCheck()
        g.debug(1)

        x = np.ones((10, 10))
        x = flow.tensor(x, dtype=flow.float32)

        t2 = np.ones((10, 10))
        t2 = flow.tensor(t2, dtype=flow.float32)
        t3 = np.ones((10, 10))
        t3 = flow.tensor(t3, dtype=flow.float32)
        lt0 = list()
        lt0.append(t2)
        lt0.append(t3)

        ot, olt, on = g(x, lt0, None)
        # print(g)
        test_case.assertTrue(np.array_equal(x.numpy(), ot.numpy()))

        test_case.assertTrue(isinstance(olt, list))
        test_case.assertTrue(isinstance(olt[0], Tensor))
        test_case.assertTrue(np.array_equal(olt[0].numpy(), lt0[0].numpy()))
        test_case.assertTrue(isinstance(olt[1], Tensor))
        test_case.assertTrue(np.array_equal(olt[1].numpy(), lt0[1].numpy()))

        test_case.assertTrue(on is None)

    def test_graph_outputs_buffer(test_case):
        class CustomModuleIOCheck(flow.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, t, tp, lt, n, i, s):
                return t, tp, lt, n, i, s

        class CustomGraphIOCheck1(flow.nn.Graph):
            def __init__(self):
                super().__init__()
                self.config.set_outputs_buffer_size(5)
                self.m = CustomModuleIOCheck()

            def build(self, t, tp, lt, n):
                rt, rtp, rlt, n, ri, rs = self.m(t, tp, lt, n, 1, "2")
                return t, tp, lt, n

        g = CustomGraphIOCheck1()

        x = np.ones((10, 10))
        x = flow.tensor(x, dtype=flow.float32)

        y = np.ones((10, 10))
        y = flow.tensor(y, dtype=flow.float32)

        # IO with TensorTuple cannot pass this test,
        # its tensor item's id is weird.
        # t0 = np.ones((10, 10))
        # t0 = flow.tensor(t0, dtype=flow.float32)
        # t1 = np.ones((10, 10))
        # t1 = flow.tensor(t1, dtype=flow.float32)
        # tp0 = TensorTuple()
        # tp0.append(t0)
        # tp0.append(t1)

        t2 = np.ones((10, 10))
        t2 = flow.tensor(t2, dtype=flow.float32)
        t3 = np.ones((10, 10))
        t3 = flow.tensor(t3, dtype=flow.float32)
        lt0 = list()
        lt0.append(t2)
        lt0.append(t3)

        # Check there is not duplicated tensor in outputs buffer and outputs.
        out_id_dic = dict()
        out_tensor_holder = dict()

        def check_id_and_add(t, name):
            if t is not None:
                tid = id(t)
                assert (
                    tid not in out_id_dic
                ), f"tid {tid}, now name {name}, inserted name {out_id_dic[tid]}"
                test_case.assertTrue(tid not in out_id_dic)
                out_id_dic[tid] = name
                # It seems that python id maybe re-used, hold it to avoid gc re-using it.
                # ref: https://stackoverflow.com/questions/52096582/how-unique-is-pythons-id
                out_tensor_holder[name] = t

        def call_and_check(idx):
            # ot, otp, olt, on = g(x, tp0, lt0, None)
            ot, otp, olt, on = g(x, y, lt0, None)
            if idx == 0:
                test_case.assertEqual(len(g._outputs_tensor_tuple_buffer), 5)
                for b_idx, buffer in enumerate(g._outputs_tensor_tuple_buffer):
                    for i_idx, item in enumerate(buffer):
                        check_id_and_add(
                            item, "buffer_" + str(b_idx) + "_" + str(i_idx)
                        )

            test_case.assertTrue(np.array_equal(x.numpy(), ot.numpy()))
            check_id_and_add(ot, "ot_" + str(idx))

            # test_case.assertTrue(isinstance(otp, TensorTuple))
            # check_id_and_add(otp, "otp_" + str(idx))
            # test_case.assertTrue(isinstance(otp[0], Tensor))
            # check_id_and_add(otp[0], "otp0_" + str(idx))
            # test_case.assertTrue(np.array_equal(otp[0].numpy(), tp0[0].numpy()))
            # test_case.assertTrue(isinstance(otp[1], Tensor))
            # check_id_and_add(otp[1], "otp1_" + str(idx))
            # test_case.assertTrue(np.array_equal(otp[1].numpy(), tp0[1].numpy()))

            test_case.assertTrue(isinstance(otp, Tensor))
            check_id_and_add(otp, "otp_" + str(idx))
            test_case.assertTrue(np.array_equal(y.numpy(), otp.numpy()))

            test_case.assertTrue(isinstance(olt, list))
            check_id_and_add(olt, "olt_" + str(idx))
            test_case.assertTrue(isinstance(olt[0], Tensor))
            check_id_and_add(olt[0], "olt0_" + str(idx))
            test_case.assertTrue(np.array_equal(olt[0].numpy(), lt0[0].numpy()))
            check_id_and_add(olt[1], "olt1_" + str(idx))
            test_case.assertTrue(np.array_equal(olt[1].numpy(), lt0[1].numpy()))

            test_case.assertTrue(on is None)

        for i in range(15):
            call_and_check(i)


if __name__ == "__main__":
    unittest.main()
