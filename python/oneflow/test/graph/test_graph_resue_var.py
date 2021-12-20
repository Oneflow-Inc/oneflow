import os
import unittest
import numpy as np
import oneflow as flow
import oneflow.unittest
@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
@flow.unittest.skip_unless_1n2d()
class TestConsistentAsymmetricGraph(oneflow.unittest.TestCase):
    def test_consistent_asymmetric_graph_gpu(test_case):
        Broadcast = flow.sbp.broadcast
        Placement_rank_0 = flow.placement("cuda", {0: [0]})
        Placement_rank_1 = flow.placement("cuda", {0: [1]})
        class MyConsistentAsymmetricModule(flow.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = flow.nn.Linear(3, 4, False)
                self.linear2 = flow.nn.Linear(4, 4, False)
                self.linear1.to_consistent(placement=Placement_rank_0, sbp=Broadcast)
                self.linear2.to_consistent(placement=Placement_rank_1, sbp=Broadcast)
                flow.nn.init.ones_(self.linear1.weight)
                flow.nn.init.constant_(self.linear2.weight, 2.3)
                self.duplicate_weight = self.linear2.weight
                self.linear3 = flow.nn.Linear(4, 4, False)
                self.linear3.to_consistent(placement=Placement_rank_1, sbp=Broadcast)
            def forward(self, x, y):
                out0 = x + y
                out0 = out0.to_consistent(
                        placement=flow.placement("cuda", {0:[0, 1]}),
                        sbp=Broadcast)
                print("cclog: out0 1d = ", out0)
                out0 = out0.to_consistent(
                        placement=flow.placement("cuda", {0:[0, 1]}, hierarchy=(1, 2)),
                        sbp=[Broadcast, Broadcast])
                print("cclog: out0 2d = ", out0)
                out0 = out0.to_consistent(placement=Placement_rank_0,
                        sbp=Broadcast)
                out1 = self.linear1(out0)
                out1 = out1.to_consistent(placement=Placement_rank_1, sbp=Broadcast)
                out2 = self.linear2(out1)
                out3 = flow._C.matmul(out2, self.duplicate_weight)
                out4 = self.linear3(out3)
                return out4
        class MyLocalModule(flow.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = flow.nn.Linear(3, 4, False)
                self.linear2 = flow.nn.Linear(4, 4, False)
                flow.nn.init.ones_(self.linear1.weight)
                flow.nn.init.constant_(self.linear2.weight, 2.3)
                self.duplicate_weight = self.linear2.weight
                self.linear3 = flow.nn.Linear(4, 4, False)
            def forward(self, x, y):
                # print("local_x in rank : ", flow.env.get_rank(), " is : ", x)
                # print("local_y in rank : ", flow.env.get_rank(), " is : ", y)
                out0 = x + y
                out1 = self.linear1(out0)
                out2 = self.linear2(out1)
                out3 = flow._C.matmul(out2, self.duplicate_weight)
                out4 = self.linear3(out3)
                return out4
        my_local_module = MyLocalModule()
        my_local_module.linear3.weight = my_local_module.linear2.weight
        np_x = np.random.randn(5, 3)
        np_y = np.ones(3)
        local_x = flow.tensor(np_x, dtype=flow.float32)
        consistent_x = local_x.to_consistent(
            placement=flow.placement("cuda", {0: [0, 1]}), sbp=Broadcast
        )
        local_x = consistent_x.to_local().to("cpu")
        local_y = flow.tensor(np_y, dtype=flow.float32)
        local_out = my_local_module(local_x, local_y)
        # print("eager_local_out: ", local_out)
        my_module = MyConsistentAsymmetricModule()
        x = local_x.to_consistent(placement=Placement_rank_0, sbp=Broadcast)
        y = local_y.to_consistent(placement=Placement_rank_0, sbp=Broadcast)
        my_module.linear3.weight = my_module.linear2.weight
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