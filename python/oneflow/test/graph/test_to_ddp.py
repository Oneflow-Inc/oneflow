import unittest
import os
import numpy as np

import oneflow as flow
import oneflow.unittest
from oneflow.nn.graph import GraphModule


def _test_linear_train_graph_with_ddp(test_case):
    def train_with_graph(iter_num=1):
        PC = flow.placement("cpu", ranks=[0, 1])
        P = flow.placement("cuda", ranks=[0, 1])
        B = flow.sbp.broadcast
        S0 = flow.sbp.split(0)

        linear_dp = flow.nn.Linear(800, 400, bias=False)
        linear_dp = linear_dp.to_global(placement=P, sbp=B)
        flow.nn.init.constant_(linear_dp.weight, 2.068758)

        of_sgd = flow.optim.SGD(
            [{"params": linear_dp.parameters()}],
            lr=0.001,
            momentum=0.9,
        )

        x = flow.ones((6, 800), placement=PC, sbp=S0)

        class LinearTrainGraphWithDDP(flow.nn.Graph):
            def __init__(self):
                super().__init__()
                self.linear_dp = linear_dp
                self.add_optimizer(of_sgd)

            def build(self, x):
                x = x.to_global(placement=P)
                out = self.linear_dp(x)
                loss = out.sum()
                loss.backward()
                return out

        class LinearEvalGraphWithDDP(flow.nn.Graph):
            def __init__(self):
                super().__init__()
                self.linear_dp = linear_dp

            def build(self, x):
                x = x.to_global(placement=P)
                out = self.linear_dp(x)
                return out

        linear_t_g = LinearTrainGraphWithDDP()
        #linear_t_g.debug(1)
        linear_e_g = LinearEvalGraphWithDDP()
        #linear_e_g.debug(1)

        result_check_list = []
        def one_train_iter(iter_cnt=0):
            out = linear_t_g(x)
            result_check_list.append(out)

            if iter_cnt == 0:
                if flow.env.get_rank() == 0:
                    import traceback

                    try:
                        print(linear_t_g)
                    except:
                        print(traceback.format_exc())

        def one_eval_iter(iter_cnt=0):
            out = linear_e_g(x)
            result_check_list.append(out)

        for i in range(iter_num):
            one_train_iter(i)

        # In evaluation graph, paramters's sbp are flow.sbp.split(0).
        # But their consumer will consum them as flow.sbp.broadcast.
        one_eval_iter()

        return result_check_list

    def train_with_graph_ddp(iter_num=1):
        PC = flow.placement("cpu", ranks=[0, 1])
        P = flow.placement("cuda", ranks=[0, 1])
        B = flow.sbp.broadcast
        S0 = flow.sbp.split(0)

        linear_dp = flow.nn.Linear(800, 400, bias=False)
        linear_dp = linear_dp.to_global(placement=P, sbp=B)
        flow.nn.init.constant_(linear_dp.weight, 2.068758)

        of_sgd = flow.optim.SGD(
            [{"params": linear_dp.parameters()}],
            lr=0.001,
            momentum=0.9,
        )

        x = flow.ones((6, 800), placement=PC, sbp=S0)

        class LinearTrainGraphWithDDP(flow.nn.Graph):
            def __init__(self):
                super().__init__()
                self.linear_dp = linear_dp
                self.add_optimizer(of_sgd)

            def build(self, x):
                x = x.to_global(placement=P)
                out = self.linear_dp(x)
                loss = out.sum()
                loss.backward()
                return out

        class LinearEvalGraphWithDDP(flow.nn.Graph):
            def __init__(self):
                super().__init__()
                self.linear_dp = linear_dp

            def build(self, x):
                x = x.to_global(placement=P)
                out = self.linear_dp(x)
                return out

        linear_t_g = LinearTrainGraphWithDDP()
        #linear_t_g.debug(1)
        linear_e_g = LinearEvalGraphWithDDP()
        #linear_e_g.debug(1)

        result_check_list = []
        def one_train_iter(iter_cnt=0):
            out = linear_t_g(x)
            result_check_list.append(out)

            if iter_cnt == 0:
                if flow.env.get_rank() == 0:
                    import traceback

                    try:
                        print(linear_t_g)
                    except:
                        print(traceback.format_exc())

        def one_eval_iter(iter_cnt=0):
            out = linear_e_g(x)
            result_check_list.append(out)

        for i in range(iter_num):
            one_train_iter(i)

        # In evaluation graph, paramters's sbp are flow.sbp.split(0).
        # But their consumer will consum them as flow.sbp.broadcast.
        one_eval_iter()

        return result_check_list

    iter_num = 2
    graph_check_list = train_with_graph(iter_num)
    graph_ddp_check_list = train_with_graph_ddp(iter_num)
    test_case.assertEqual(len(graph_check_list), iter_num + 1)
    test_case.assertEqual(len(graph_ddp_check_list), iter_num + 1)
    for i in range(iter_num + 1):
        test_case.assertTrue(np.array_equal(graph_check_list[i].numpy(), graph_ddp_check_list[i].numpy()));

@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
@flow.unittest.skip_unless_1n2d()
class TestLinearTrainGraphWithDDP(oneflow.unittest.TestCase):
    def test_linear_train_graph_with_ddp(test_case):
        _test_linear_train_graph_with_ddp(test_case)

if __name__ == "__main__":
    unittest.main()