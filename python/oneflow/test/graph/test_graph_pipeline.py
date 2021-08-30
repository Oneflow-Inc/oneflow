import os
import sys

# For debug
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "8003"
os.environ["WORLD_SIZE"] = "2"
os.environ["RANK"] = str(sys.argv[1])
os.environ["LOCAL_RANK"] = str(sys.argv[1])

import unittest
import numpy as np

import oneflow as flow
import oneflow.unittest


def _test_train_graph(test_case, device):
    rank = flow.env.get_rank()
    def train_with_module(iter_num=3):
        class LocalModule(flow.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear0 = flow.nn.Linear(3, 8, False)
                self.linear1 = flow.nn.Linear(8, 7, False)
                flow.nn.init.ones_(self.linear0.weight)
                flow.nn.init.constant_(self.linear1.weight, 2.3)

            def forward(self, x):
                out0 = self.linear0(x)
                out1 = self.linear1(out0)
                return out1

        local_m = LocalModule()
        local_m = local_m.to(device)

        of_sgd = flow.optim.SGD(local_m.parameters(), lr=0.001, momentum=0.9)

        x = flow.Tensor(
            [
                [-0.94630778, -0.83378579, -0.87060891],
                [2.0289922, -0.28708987, -2.18369248],
                [0.35217619, -0.67095644, -1.58943879],
                [0.08086036, -1.81075924, 1.20752494],
                [0.8901075, -0.49976737, -1.07153746],
                [-0.44872912, -1.07275683, 0.06256855],
                [-0.22556897, 0.74798368, 0.90416439],
                [0.48339456, -2.32742195, -0.59321527],
            ],
            device=device,
            requires_grad=False,
        )

        def one_iter():
            of_out = local_m(x)
            of_out = of_out.sum()

            of_out.backward()
            of_sgd.step()
            of_sgd.zero_grad()
            
            print("rank: ", rank, " eager out:", of_out.numpy())
            return of_out.numpy(), local_m.linear1.weight.numpy()

        check_list = []
        for i in range(iter_num):
            check_list.append(one_iter())
        return check_list

    def train_with_graph(iter_num=3):
        B = [flow.sbp.broadcast]
        P = flow.placement("cuda", {0: [0, 1]})
        P0 = flow.placement("cuda", {0: [0]})
        P1 = flow.placement("cuda", {0: [1]})

        class PipelineModule(flow.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear0 = flow.nn.Linear(3, 8, False)
                self.linear1 = flow.nn.Linear(8, 7, False)
                self.linear0.to_consistent(placement=P0, sbp=B)
                self.linear1.to_consistent(placement=P1, sbp=B)
                flow.nn.init.ones_(self.linear0.weight)
                flow.nn.init.constant_(self.linear1.weight, 2.3)

            def forward(self, x):
                out0 = self.linear0(x)
                out0 = out0.to_consistent(placement=P1, sbp=B)
                out1 = self.linear1(out0)
                return out1

        pp_m = PipelineModule()

        of_sgd = flow.optim.SGD(pp_m.parameters(), lr=0.001, momentum=0.9)

        class PipelineGraph(flow.nn.Graph):
            def __init__(self):
                super().__init__()
                self.pp_m = pp_m
                self.pp_m.linear0.stage_id = 0
                self.pp_m.linear1.stage_id = 1
                # TODO(): support gradient accumulation
                #self.config.set_gradient_accumulation_steps(3)
                self.add_optimizer(of_sgd)

            def build(self, x):
                out = self.pp_m(x)
                out = out.sum()
                # TODO(): support partial placement of scalar tensor numpy()
                out = out.to_consistent(placement=P, sbp=B)
                out.backward()
                return out

        pp_g = PipelineGraph()

        x = flow.Tensor(
            [
                [-0.94630778, -0.83378579, -0.87060891],
                [2.0289922, -0.28708987, -2.18369248],
                [0.35217619, -0.67095644, -1.58943879],
                [0.08086036, -1.81075924, 1.20752494],
                [0.8901075, -0.49976737, -1.07153746],
                [-0.44872912, -1.07275683, 0.06256855],
                [-0.22556897, 0.74798368, 0.90416439],
                [0.48339456, -2.32742195, -0.59321527],
            ],
            device=device,
            requires_grad=False,
        )
        x = x.to_consistent(placement=P0, sbp=B)

        def one_iter():
            pp_m.train()
            of_graph_out = pp_g(x)
            test_case.assertTrue(of_graph_out.placement == P)
            of_graph_out = of_graph_out.to_local()
            of_graph_out_np = of_graph_out.numpy()
            print("rank: ", rank, " pipeline graph out: ", of_graph_out_np)
            return of_graph_out_np, pp_m.linear1.weight.to_local().numpy()

        check_list = []
        for i in range(iter_num):
            check_list.append(one_iter())
        return check_list

    iter_num = 2
    if (rank == 1):
        module_check_list = train_with_module(iter_num)

    graph_check_list = train_with_graph(iter_num)

    if (rank == 1):
        for i in range(iter_num):
            # check equal on loss
            test_case.assertTrue(
                np.array_equal(module_check_list[i][0], graph_check_list[i][0])
            )
            # check equal on weight
            test_case.assertTrue(
                np.array_equal(module_check_list[i][1], graph_check_list[i][1])
            )


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
@flow.unittest.skip_unless_1n1d()
class TestGraphPipeline(oneflow.unittest.TestCase):
    def test_train_graph_gpu(test_case):
        _test_train_graph(test_case, flow.device("cuda"))


if __name__ == "__main__":
    sys.argv.pop()
    unittest.main()