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


def _test_linear_train_graph_with_zero(test_case, zero_stage=1):
    def train_with_graph(iter_num=1):
        P = flow.placement("cuda", ranks=[0, 1])
        B = flow.sbp.broadcast
        S0 = flow.sbp.split(0)

        linear_dp = flow.nn.Linear(800, 400, bias=False)
        linear_dp = linear_dp.to_global(placement=P, sbp=B)
        flow.nn.init.constant_(linear_dp.weight, 2.068758)

        linear_mp = flow.nn.Linear(400, 500, bias=False)
        linear_mp = linear_mp.to_global(placement=P, sbp=S0)
        flow.nn.init.constant_(linear_mp.weight, 2.068758)

        of_sgd = flow.optim.SGD(
            [{"params": linear_dp.parameters()}, {"params": linear_mp.parameters()}],
            lr=0.001,
            momentum=0.9,
        )
        grad_scaler = flow.amp.StaticGradScaler(200)

        x = flow.randint(1, 100, (6, 800), dtype=flow.float32, placement=P, sbp=S0)

        class LinearTrainGraphWithZeRO(flow.nn.Graph):
            def __init__(self):
                super().__init__()
                self.linear_dp = linear_dp
                self.linear_mp = linear_mp
                self.add_optimizer(of_sgd)

                self.config.enable_amp(True)
                self.set_grad_scaler(grad_scaler)
                self.config.enable_zero(
                    True, stage=zero_stage, shard_min_size=1, shard_restore_level=0,
                )
                self.debug(2)

            def build(self, x):
                out = self.linear_dp(x)
                out = out.to_global(placement=P, sbp=B)
                out = self.linear_mp(out)
                loss = out.sum()
                loss.backward()
                return out

        class LinearEvalGraphWithZeRO(flow.nn.Graph):
            def __init__(self):
                super().__init__()
                self.linear_dp = linear_dp
                self.linear_mp = linear_mp

                self.config.enable_amp(True)

            def build(self, x):
                out = self.linear_dp(x)
                out = out.to_global(placement=P, sbp=B)
                out = self.linear_mp(out)
                return out

        linear_t_g = LinearTrainGraphWithZeRO()
        linear_t_g.debug(1)
        linear_e_g = LinearEvalGraphWithZeRO()
        linear_e_g.debug(1)

        def one_train_iter():
            out = linear_t_g(x)
            if flow.env.get_rank() == 0:
                import traceback

                try:
                    print(linear_t_g)
                except:
                    print(traceback.format_exc())

        def one_eval_iter():
            out = linear_e_g(x)

        for i in range(iter_num):
            one_train_iter()

        # After pass rewrite in training graph, parameters' sbp has been
        # changed from flow.sbp.broadcast to flow.sbp.split(0)
        test_case.assertEqual(linear_dp.weight.sbp[0], S0)
        test_case.assertEqual(linear_mp.weight.sbp[0], S0)

        # In evaluation graph, paramters's sbp are flow.sbp.split(0).
        # But their consumer will consum them as flow.sbp.broadcast.
        one_eval_iter()

    iter_num = 1
    graph_check_list = train_with_graph(iter_num)


def _test_linear_train_graph_2d_with_zero(test_case, zero_stage=1):
    def train_with_graph(iter_num=1):
        P = flow.placement("cuda", ranks=[[0, 1], [2, 3]])
        B = flow.sbp.broadcast
        S0 = flow.sbp.split(0)
        S1 = flow.sbp.split(1)

        def get_mixed_linear():
            linear_dp_mp = flow.nn.Linear(800, 400, bias=False)
            linear_dp_mp = linear_dp_mp.to_global(placement=P, sbp=[B, S0])
            flow.nn.init.constant_(linear_dp_mp.weight, 1.068758)

            linear_mp_dp = flow.nn.Linear(800, 400, bias=False)
            linear_mp_dp = linear_mp_dp.to_global(placement=P, sbp=[S0, B])
            flow.nn.init.constant_(linear_mp_dp.weight, 1.068758)

            class MixedLinear(flow.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.dp_mp = linear_dp_mp
                    self.mp_dp = linear_mp_dp

                def forward(self, x):
                    x = self.dp_mp(x)
                    x = flow.relu(x)
                    x = self.mp_dp(x)
                    x = flow.relu(x)
                    return x

            return MixedLinear()

        mixed_linear0 = get_mixed_linear()
        mixed_linear1 = get_mixed_linear()

        of_sgd = flow.optim.SGD(
            [
                {"params": mixed_linear0.parameters()},
                {"params": mixed_linear1.parameters()},
            ],
            lr=0.001,
            momentum=0.9,
        )
        grad_scaler = flow.amp.StaticGradScaler(200)

        x = flow.rand((2, 800), dtype=flow.float32, placement=P, sbp=[S0, B])

        class LinearTrainGraph2DWithZeRO(flow.nn.Graph):
            def __init__(self):
                super().__init__()
                self.mixed_linear0 = mixed_linear0
                self.mixed_linear0.to(GraphModule).activation_checkpointing = True
                self.mixed_linear1 = mixed_linear1
                self.mixed_linear1.to(GraphModule).activation_checkpointing = True
                self.add_optimizer(of_sgd)

                self.config.enable_amp(True)
                self.set_grad_scaler(grad_scaler)
                self.config.enable_zero(
                    True, stage=zero_stage, shard_min_size=1, shard_restore_level=1,
                )

            def build(self, x):
                out = self.mixed_linear0(x)
                out = self.mixed_linear1(out)
                loss = out.mean()
                loss.backward()
                return loss

        class LinearEvalGraph2DWithZeRO(flow.nn.Graph):
            def __init__(self):
                super().__init__()
                self.mixed_linear0 = mixed_linear0
                self.mixed_linear1 = mixed_linear1

                self.config.enable_amp(True)

            def build(self, x):
                out = self.mixed_linear0(x)
                out = self.mixed_linear1(out)
                return out

        linear_t_g = LinearTrainGraph2DWithZeRO()
        linear_e_g = LinearEvalGraph2DWithZeRO()

        def one_train_iter():
            out = linear_t_g(x)
            # if flow.env.get_rank() == 0:
            #    print(linear_t_g)

        def one_eval_iter():
            out = linear_e_g(x)

        for i in range(iter_num):
            one_train_iter()

        for state in linear_t_g._state():
            test_case.assertEqual(
                state.to(flow.Tensor).sbp,
                (oneflow.sbp.split(dim=0), oneflow.sbp.split(dim=0)),
            )

        # In evaluation graph, paramters's sbp are flow.sbp.split(0).
        # But their consumer will consum them as flow.sbp.broadcast.
        one_eval_iter()

    iter_num = 1
    graph_check_list = train_with_graph(iter_num)


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
@flow.unittest.skip_unless_1n2d()
class TestLinearTrainGraphWithZeRO(oneflow.unittest.TestCase):
    def test_linear_train_graph_with_zero_1(test_case):
        _test_linear_train_graph_with_zero(test_case, 1)

    def test_linear_train_graph_with_zero_2(test_case):
        _test_linear_train_graph_with_zero(test_case, 2)

    def test_linear_train_graph_with_zero_3(test_case):
        _test_linear_train_graph_with_zero(test_case, 3)


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
@flow.unittest.skip_unless_1n4d()
class TestLinearTrainGraph2DWithZeRO(oneflow.unittest.TestCase):
    def test_linear_train_graph_2d_with_zero_3(test_case):
        _test_linear_train_graph_2d_with_zero(test_case, 3)

    def test_linear_train_graph_2d_with_zero_2(test_case):
        _test_linear_train_graph_2d_with_zero(test_case, 2)

    def test_linear_train_graph_2d_with_zero_1(test_case):
        _test_linear_train_graph_2d_with_zero(test_case, 1)


if __name__ == "__main__":
    unittest.main()
