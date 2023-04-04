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
import tempfile

import oneflow as flow
import oneflow.unittest


def _test_linear_graph_save_load(test_case, device):
    def train_with_graph(call_cnt=0, state_dict_file=None, last_state_dict=None):
        linear = flow.nn.Linear(3, 8)
        linear = linear.to(device)
        flow.nn.init.constant_(linear.weight, 2.068758)
        flow.nn.init.constant_(linear.bias, 0.23)
        of_sgd = flow.optim.SGD(linear.parameters(), lr=0.001, momentum=0.9)

        x = flow.tensor(
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
            dtype=flow.float32,
            device=device,
            requires_grad=False,
        )

        class LinearTrainGraph(flow.nn.Graph):
            def __init__(self):
                super().__init__()
                self.linear = linear
                self.add_optimizer(of_sgd)

            def build(self, x):
                out = self.linear(x)
                out = out.sum()
                out.backward()
                return out

        linear_t_g = LinearTrainGraph()
        if call_cnt == 1:
            state_dict = flow.load(state_dict_file)
            linear_t_g.load_state_dict(state_dict)
            # Check state in module has been loaded.
            test_case.assertTrue(
                np.array_equal(state_dict["linear"]["weight"].numpy(), linear.weight)
            )
            test_case.assertTrue(
                np.array_equal(state_dict["linear"]["bias"].numpy(), linear.bias)
            )
        # Get state dict before compile is allowed.
        init_state_dict = linear_t_g.state_dict()

        of_graph_out = linear_t_g(x)
        iter0_state_dict = linear_t_g.state_dict()
        if call_cnt == 1:
            # Check additional variable state initialized in job has been loaded.
            cur_train_step = iter0_state_dict["System-Train-TrainStep"].numpy()[0]
            test_case.assertEqual(3, cur_train_step)
            test_case.assertTrue(
                cur_train_step == last_state_dict["System-Train-TrainStep"].numpy()[0]
            )
            test_case.assertTrue(
                np.array_equal(
                    iter0_state_dict["linear"]["weight"].numpy(),
                    last_state_dict["linear"]["weight"].numpy(),
                )
            )
            test_case.assertTrue(
                np.array_equal(
                    iter0_state_dict["linear"]["bias"].numpy(),
                    last_state_dict["linear"]["bias"].numpy(),
                )
            )
            test_case.assertTrue(
                np.array_equal(
                    iter0_state_dict["linear.weight-momentum"].numpy(),
                    last_state_dict["linear.weight-momentum"].numpy(),
                )
            )
            test_case.assertTrue(
                np.array_equal(
                    iter0_state_dict["linear.bias-momentum"].numpy(),
                    last_state_dict["linear.bias-momentum"].numpy(),
                )
            )

        of_graph_out = linear_t_g(x)
        of_graph_out.numpy()
        iter1_state_dict = linear_t_g.state_dict()
        if call_cnt == 0:
            flow.save(iter1_state_dict, state_dict_file)

        if call_cnt == 0:
            of_graph_out = linear_t_g(x)
            iter2_state_dict = linear_t_g.state_dict()
            of_graph_out.numpy()
            return iter2_state_dict

    with tempfile.NamedTemporaryFile(prefix="graph_save_load_local") as f:
        iter2_state_dict = train_with_graph(0, f.name)
        train_with_graph(1, f.name, iter2_state_dict)


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
@flow.unittest.skip_unless_1n1d()
class TestLinearGraphSaveLoad(oneflow.unittest.TestCase):
    def test_linear_graph_save_load_gpu(test_case):
        _test_linear_graph_save_load(test_case, flow.device("cuda"))

    def _test_linear_graph_save_load_cpu(test_case):
        _test_linear_graph_save_load(test_case, flow.device("cpu"))


def _test_linear_graph_save_load_global(test_case, device):
    P = flow.placement("cuda", ranks=[0, 1])
    B = flow.sbp.broadcast
    S = flow.sbp.split(0)

    def train_with_graph(call_cnt=0, state_dict_file=None, last_state_dict=None):
        linear = flow.nn.Linear(3, 8)
        linear = linear.to(device)
        flow.nn.init.constant_(linear.weight, 2.068758)
        flow.nn.init.constant_(linear.bias, 0.23)
        linear.to_global(placement=P, sbp=B)
        of_sgd = flow.optim.SGD(linear.parameters(), lr=0.001, momentum=0.9)

        x = flow.tensor(
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
            dtype=flow.float32,
            device=device,
            requires_grad=False,
        )
        x = x.to_global(placement=P, sbp=S)

        class LinearTrainGraphGlobal(flow.nn.Graph):
            def __init__(self):
                super().__init__()
                self.linear = linear
                self.add_optimizer(of_sgd)

            def build(self, x):
                out = self.linear(x)
                out = out.sum()
                out.backward()
                return out

        linear_t_g = LinearTrainGraphGlobal()
        if call_cnt == 1:
            state_dict = flow.load(state_dict_file, global_src_rank=0)
            linear_t_g.load_state_dict(state_dict)
            # Check state in module has been loaded.
            # Tensors in state dict are save to rank 0, so they need to be broadcast to rank 0 and 1 before check.
            test_case.assertTrue(
                np.array_equal(
                    state_dict["linear"]["weight"]
                    .to_global(placement=P, sbp=B)
                    .to_local()
                    .numpy(),
                    linear.weight.to_local().numpy(),
                )
            )
            test_case.assertTrue(
                np.array_equal(
                    state_dict["linear"]["bias"]
                    .to_global(placement=P, sbp=B)
                    .to_local()
                    .numpy(),
                    linear.bias.to_local().numpy(),
                )
            )
        # Get state dict before compile is allowed.
        init_state_dict = linear_t_g.state_dict()

        of_graph_out = linear_t_g(x)
        iter0_state_dict = linear_t_g.state_dict()
        if call_cnt == 1:
            # Check additional variable state initialized in job has been loaded.
            # TrainStep's placement is only on rank 0, so it needs to be broadcast to rank 0 and 1 before check.
            cur_train_step = (
                iter0_state_dict["System-Train-TrainStep"]
                .to_global(placement=P, sbp=B)
                .to_local()
                .numpy()[0]
            )
            test_case.assertEqual(3, cur_train_step)
            test_case.assertTrue(
                cur_train_step
                == last_state_dict["System-Train-TrainStep"]
                .to_global(placement=P, sbp=B)
                .to_local()
                .numpy()[0]
            )
            test_case.assertTrue(
                np.array_equal(
                    iter0_state_dict["linear"]["weight"].to_local().numpy(),
                    last_state_dict["linear"]["weight"].to_local().numpy(),
                )
            )
            test_case.assertTrue(
                np.array_equal(
                    iter0_state_dict["linear"]["bias"].to_local().numpy(),
                    last_state_dict["linear"]["bias"].to_local().numpy(),
                )
            )
            test_case.assertTrue(
                np.array_equal(
                    iter0_state_dict["linear.weight-momentum"].to_local().numpy(),
                    last_state_dict["linear.weight-momentum"].to_local().numpy(),
                )
            )
            test_case.assertTrue(
                np.array_equal(
                    iter0_state_dict["linear.bias-momentum"].to_local().numpy(),
                    last_state_dict["linear.bias-momentum"].to_local().numpy(),
                )
            )

        of_graph_out = linear_t_g(x)
        of_graph_out.numpy()
        iter1_state_dict = linear_t_g.state_dict()
        if call_cnt == 0:
            flow.save(iter1_state_dict, state_dict_file, global_dst_rank=0)

        if call_cnt == 0:
            of_graph_out = linear_t_g(x)
            of_graph_out.numpy()
            iter2_state_dict = linear_t_g.state_dict()
            return iter2_state_dict

    with tempfile.NamedTemporaryFile(prefix="graph_save_load_global") as f:
        iter2_state_dict = train_with_graph(0, f.name)
        train_with_graph(1, f.name, iter2_state_dict)


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
@flow.unittest.skip_unless_1n2d()
class TestLinearGraphSaveLoadGlobal(oneflow.unittest.TestCase):
    def test_linear_graph_save_load_gpu(test_case):
        _test_linear_graph_save_load_global(test_case, flow.device("cuda"))

    def _test_linear_graph_save_load_cpu(test_case):
        _test_linear_graph_save_load_global(test_case, flow.device("cpu"))


if __name__ == "__main__":
    unittest.main()
