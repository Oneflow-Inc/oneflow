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
import os
import numpy as np

import oneflow as flow
import oneflow.unittest
from oneflow.nn.graph import GraphModule
import oneflow.utils.global_view as global_view
from oneflow.utils.global_view import global_mode


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
            [{"params": linear_dp.parameters()}], lr=0.001, momentum=0.9,
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
        # linear_t_g.debug(1)
        linear_e_g = LinearEvalGraphWithDDP()
        # linear_e_g.debug(1)

        result_check_list = []

        def one_train_iter(iter_cnt=0):
            out = linear_t_g(x)
            result_check_list.append(out)

            # if iter_cnt == 0:
            #     if flow.env.get_rank() == 0:
            #         import traceback

            #         try:
            #             print(linear_t_g)
            #         except:
            #             print(traceback.format_exc())

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
            [{"params": linear_dp.parameters()}], lr=0.001, momentum=0.9,
        )

        with global_mode(True, placement=PC, sbp=S0):
            x = flow.ones((6, 800), placement=PC, sbp=S0)

        class LinearTrainGraphWithDDP(flow.nn.Graph):
            def __init__(self):
                super().__init__()
                self.linear_dp = linear_dp
                self.add_optimizer(of_sgd)

            def build(self, x):
                # This is ok
                # x = x.to("cuda")

                # This is ok
                # x = x.to_global(placement=P)

                # This is not ok
                # x = x.to(device)

                with global_mode(True, placement=P, sbp=B):
                    # Test global tensor to device
                    device = self.linear_dp.weight.device

                    x = x.to(device)

                    out = self.linear_dp(x)

                    # Test randn source op
                    sample = flow.randn(out.shape, device="cpu").to(device)
                    out = out + sample * 100

                # Test disable global_mode while passing placement and sbp
                with global_mode(False, placement=P, sbp=B):
                    out = out - sample * 100
                    cur_global_mode = global_view.current_global_mode()
                    test_case.assertFalse(cur_global_mode.is_enabled)

                loss = out.sum()
                loss.backward()
                return out

        class LinearEvalGraphWithDDP(flow.nn.Graph):
            def __init__(self):
                super().__init__()
                self.linear_dp = linear_dp

            def build(self, x):
                with global_mode(True, placement=P, sbp=B):
                    device = self.linear_dp.weight.device

                    x = x.to(device)

                    out = self.linear_dp(x)

                    # Test randn source op
                    sample = flow.randn(out.shape, device="cpu").to(device)
                    out = out + sample * 100
                    out = out - sample * 100

                return out

        linear_t_g = LinearTrainGraphWithDDP()
        # linear_t_g.debug(1)
        linear_e_g = LinearEvalGraphWithDDP()
        # linear_e_g.debug(1)

        result_check_list = []

        def one_train_iter(iter_cnt=0):
            out = linear_t_g(x)
            result_check_list.append(out)

            # if iter_cnt == 0:
            #     if flow.env.get_rank() == 0:
            #         import traceback

            #         try:
            #             print(linear_t_g)
            #         except:
            #             print(traceback.format_exc())

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
        test_case.assertTrue(
            np.allclose(
                graph_check_list[i].numpy(),
                graph_ddp_check_list[i].numpy(),
                rtol=1e-5,
                atol=1e-5,
            ),
            f"current index {i} \n base {graph_check_list[i].numpy()} \n ddp {graph_ddp_check_list[i].numpy()} \n diff {graph_ddp_check_list[i].numpy() - graph_check_list[i].numpy()}",
        )


def _test_global_mode(test_case):
    P = flow.placement("cuda", ranks=[0, 1])
    B = flow.sbp.broadcast

    class GlobalModeGraph(flow.nn.Graph):
        def __init__(self):
            super().__init__()

        def build(self):
            with global_mode(True, placement=P, sbp=B):
                # Test global mode meta data
                cur_global_mode = global_view.current_global_mode()
                test_case.assertTrue(cur_global_mode.is_enabled)
                test_case.assertEqual(cur_global_mode.placement, P)
                test_case.assertEqual(cur_global_mode.sbp[0], B)

                # Test global mode source op
                randn_out = flow.randn((2, 2))
                rand_out = flow.rand((2, 2))
                randint_out = flow.randint(-100, 100, (2, 2))
                randperm_out = flow.randperm(5)
                arange_out = flow.arange(10)
                empty_out = flow.empty((1, 2))
                tensor_out = flow.tensor([[1, 2, 4, 5], [4, 3, 2, 9]], dtype=flow.int)
                hann_window_out = flow.hann_window(8, dtype=flow.float)

            test_case.assertTrue(not global_view.current_global_mode().is_enabled)

            return {
                "randn_out": randn_out,
                "rand_out": rand_out,
                "randint_out": randint_out,
                "randperm_out": randperm_out,
                "arange_out": arange_out,
                "empty_out": empty_out,
                "tensor_out": tensor_out,
                "hann_window_out": hann_window_out,
            }

    global_graph = GlobalModeGraph()
    out = global_graph()
    for k, v in out.items():
        test_case.assertEqual(v.is_global, True, k)
        test_case.assertEqual(v.placement, P, k)
        test_case.assertEqual(v.sbp[0], B, k)


def _test_global_mode_with_default_placement_and_sbp(test_case):
    # create a tensor with broadcast split and placement on rank 0
    a = flow.randn(
        (1, 8), sbp=flow.sbp.broadcast, placement=flow.placement("cuda", ranks=[0])
    )
    # enter global mode with broadcast split and placement on 2 GPUs
    with global_mode(
        True,
        placement=flow.placement(type="cuda", ranks=[0, 1]),
        sbp=flow.sbp.broadcast,
    ):
        # check tensor placement and split
        test_case.assertTrue(a.placement == flow.placement("cuda", ranks=[0]))
        test_case.assertTrue(a.sbp == (flow.sbp.broadcast,))
        # check tensor print
        print(a)


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
@flow.unittest.skip_unless_1n2d()
class TestLinearTrainGraphWithDDP(oneflow.unittest.TestCase):
    def test_linear_train_graph_with_ddp(test_case):
        _test_linear_train_graph_with_ddp(test_case)

    def test_global_mode(test_case):
        _test_global_mode(test_case)
        _test_global_mode_with_default_placement_and_sbp(test_case)


if __name__ == "__main__":
    unittest.main()
