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


def _test_linear_train_graph_with_zero(test_case, zero_stage=1):
    def train_with_graph(iter_num=1):
        P = flow.placement("cuda", {0: [0, 1]})
        B = flow.sbp.broadcast
        S0 = flow.sbp.split(0)
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
                    self.config.set_zero_redundancy_optimizer_min_size_after_split(1)
                if zero_stage == 2:
                    self.config.set_zero_redundancy_optimizer_mode("distributed_split")
                    self.config.set_zero_redundancy_optimizer_min_size_after_split(1)
                    flow.boxing.nccl.enable_use_compute_stream(True)
                if zero_stage == 3:
                    print("zero stage 3 optimization")
                    self.config.set_zero_redundancy_optimizer_mode("distributed_split")
                    self.config.set_zero_redundancy_optimizer_min_size_after_split(1)
                    flow.boxing.nccl.enable_use_compute_stream(True)
                    flow.boxing.nccl.disable_group_boxing_by_dst_parallel(True)

            def build(self, x):
                out = self.linear(x)
                loss = out.sum()
                loss.backward()
                return out

        class LinearEvalGraphWithZeRO(flow.nn.Graph):
            def __init__(self):
                super().__init__()
                self.linear = linear

                self.config.enable_amp(True)

            def build(self, x):
                out = self.linear(x)
                return out

        linear_t_g = LinearTrainGraphWithZeRO()
        linear_e_g = LinearEvalGraphWithZeRO()

        def one_train_iter():
            out = linear_t_g(x)

        def one_eval_iter():
            out = linear_e_g(x)

        for i in range(iter_num):
            one_train_iter()

        # After pass rewrite in training graph, parameters' sbp has been
        # changed from flow.sbp.broadcast to flow.sbp.split(0)
        test_case.assertEqual(linear.weight.sbp[0], S0)
        test_case.assertEqual(linear.bias.sbp[0], S0)

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


if __name__ == "__main__":
    unittest.main()
