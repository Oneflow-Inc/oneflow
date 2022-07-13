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
import numpy as np

import oneflow as flow
import oneflow.unittest


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
@flow.unittest.skip_unless_1n2d()
class TestGlobalAsymmetricGraph(oneflow.unittest.TestCase):
    def test_global_asymmetric_graph_gpu(test_case):
        Broadcast = [flow.sbp.broadcast]
        Placement_rank_0 = flow.placement("cuda", ranks=[0])
        Placement_rank_1 = flow.placement("cuda", ranks=[1])

        class MyGlobalAsymmetricModule(flow.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = flow.nn.Linear(3, 8, False)
                self.linear2 = flow.nn.Linear(8, 7, False)
                self.linear1.to_global(placement=Placement_rank_0, sbp=Broadcast)
                self.linear2.to_global(placement=Placement_rank_1, sbp=Broadcast)
                flow.nn.init.ones_(self.linear1.weight)
                flow.nn.init.constant_(self.linear2.weight, 2.3)

            def forward(self, x, y):
                out0 = x + y
                out1 = self.linear1(out0)
                out1 = out1.to_global(placement=Placement_rank_1, sbp=Broadcast)
                out2 = self.linear2(out1)
                return out2

        class MyLocalModule(flow.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = flow.nn.Linear(3, 8, False)
                self.linear2 = flow.nn.Linear(8, 7, False)
                flow.nn.init.ones_(self.linear1.weight)
                flow.nn.init.constant_(self.linear2.weight, 2.3)

            def forward(self, x, y):
                # print("local_x in rank : ", flow.env.get_rank(), " is : ", x)
                # print("local_y in rank : ", flow.env.get_rank(), " is : ", y)
                out0 = x + y
                out1 = self.linear1(out0)
                out2 = self.linear2(out1)
                return out2

        my_local_module = MyLocalModule()
        np_x = np.random.randn(5, 3)
        np_y = np.ones(3)
        local_x = flow.tensor(np_x, dtype=flow.float32)
        global_x = local_x.to_global(
            placement=flow.placement("cuda", ranks=[0, 1]), sbp=Broadcast
        )
        local_x = global_x.to_local().to("cpu")
        local_y = flow.tensor(np_y, dtype=flow.float32)
        local_out = my_local_module(local_x, local_y)
        # print("eager_local_out: ", local_out)

        my_module = MyGlobalAsymmetricModule()
        x = local_x.to_global(placement=Placement_rank_0, sbp=Broadcast)
        y = local_y.to_global(placement=Placement_rank_0, sbp=Broadcast)

        class MyAsymmetricGraph(flow.nn.Graph):
            def __init__(self):
                super().__init__()
                self.my_net = my_module

            def build(self, x, y):
                return self.my_net(x, y)

        my_g = MyAsymmetricGraph()
        graph_out = my_g(x, y)
        test_case.assertTrue(graph_out.placement == Placement_rank_1)
        graph_local_out = graph_out.to_local()
        # NOTE(chengcheng): MUST call for each rank sync correct input copy
        graph_local_out_np = graph_local_out.numpy()
        # print("graph_local_out in rank ", flow.env.get_rank(),  " is : ", graph_local_out)
        if flow.env.get_rank() == 0:
            test_case.assertTrue(graph_local_out.shape.numel() == 0)
            test_case.assertTrue(graph_local_out_np.size == np.array([]).size)
        elif flow.env.get_rank() == 1:
            test_case.assertTrue(
                np.allclose(
                    graph_local_out.numpy(), local_out.numpy(), atol=1e-4, rtol=1e-4
                )
            )
        else:
            test_case.assertTrue(False)


if __name__ == "__main__":
    unittest.main()
