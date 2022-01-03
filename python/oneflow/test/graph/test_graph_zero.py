import unittest
import os
import numpy as np

import oneflow as flow
import oneflow.unittest


def _test_linear_train_graph_with_zero(test_case, zero_stage = 1):
    P = flow.placement("cuda", {0: [0, 1]})
    B = flow.sbp.broadcast
    S0 = flow.sbp.split(0)
    def train_with_graph(iter_num=3):
        linear = flow.nn.Linear(8, 4)
        linear = linear.to_consistent(placement=P, sbp=B)
        flow.nn.init.constant_(linear.weight, 2.068758)
        flow.nn.init.constant_(linear.bias, 0.23)
        of_sgd = flow.optim.SGD(linear.parameters(), lr=0.001, momentum=0.9)
        grad_scaler = flow.amp.StaticGradScaler(200)

        x = flow.randint(1, 100, (4, 8), dtype=flow.float32, placement=P, sbp=S0)

        class LinearTrainGraphWithZeRO(flow.nn.Graph):
            def __init__(self):
                super().__init__()
                self.linear = linear
                self.add_optimizer(of_sgd)

                self.config.enable_amp(True)
                self.set_grad_scaler(grad_scaler)
                if zero_stage == 1:
                    print("zero stage 1 optimization")
                    self.config.set_zero_redundancy_optimizer_mode("distributed_split")
                    self.config._set_zero_redundancy_optimizer_split_min_size(1)
                if zero_stage == 2:
                    print("zero stage 2 optimization")
                    self.config.set_zero_redundancy_optimizer_mode("distributed_split")
                    flow.boxing.nccl.enable_use_compute_stream(True)
                if zero_stage == 3:
                    print("zero stage 3 optimization")
                    self.config.set_zero_redundancy_optimizer_mode("distributed_split")
                    flow.boxing.nccl.enable_use_compute_stream(True)
                    flow.boxing.nccl.disable_group_boxing_by_dst_parallel(True)

            def build(self, x):
                out = self.linear(x)
                loss = out.sum()
                loss.backward()
                return out

        linear_t_g = LinearTrainGraphWithZeRO()

        def one_iter():
            of_graph_out = linear_t_g(x)
            if flow.env.get_rank() == 0:
                print("graph repr ", linear_t_g)
            print("out ", of_graph_out)
            return of_graph_out.numpy(), linear_t_g.linear.weight.origin.numpy()

        check_list = []
        for i in range(iter_num):
            check_list.append(one_iter())
        return check_list

    iter_num = 1
    graph_check_list = train_with_graph(iter_num)

@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
@flow.unittest.skip_unless_1n2d()
class TestLinearTrainGraphWithZeRO(oneflow.unittest.TestCase):
    def test_linear_train_graph_with_zero_1(test_case):
        _test_linear_train_graph_with_zero(test_case, 1)


if __name__ == "__main__":
    unittest.main()
