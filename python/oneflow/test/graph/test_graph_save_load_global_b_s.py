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
import tempfile

import oneflow as flow
import oneflow.unittest
from oneflow.nn.graph import GraphModule


def _test_linear_graph_save_load_global_broadcast(
    test_case, model_tensor_placement, model_file_placement
):
    """Data parallelism on 2 ranks.
    """
    B = flow.sbp.broadcast
    S0 = flow.sbp.split(0)

    def train_with_graph(call_cnt=0, state_dict_file=None, last_state_dict=None):
        linear = flow.nn.Linear(3, 8)
        linear = linear.to(flow.device(model_tensor_placement.type))
        flow.nn.init.constant_(linear.weight, 2.068758)
        flow.nn.init.constant_(linear.bias, 0.23)
        linear.to_global(placement=model_tensor_placement, sbp=B)
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
            device=flow.device(model_tensor_placement.type),
            requires_grad=False,
        )
        x = x.to_global(placement=model_tensor_placement, sbp=S0)

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
        cur_rank = flow.env.get_rank()
        if call_cnt == 1:
            if cur_rank in model_file_placement.ranks:
                local_state_dict = flow.load(state_dict_file)
            else:
                local_state_dict = None

            global_state_dict = flow.utils.global_view.to_global(
                local_state_dict, placement=model_file_placement, sbp=B
            )
            linear_t_g.load_state_dict(global_state_dict)

            if cur_rank == 0:  # Ignore None on rank 1
                # Check state in module has been loaded.
                test_case.assertTrue(
                    np.array_equal(
                        global_state_dict["linear"]["weight"].to_local().numpy(),
                        linear.weight.to_local().numpy(),
                    )
                )
                test_case.assertTrue(
                    np.array_equal(
                        global_state_dict["linear"]["bias"].to_local().numpy(),
                        linear.bias.to_local().numpy(),
                    )
                )
        # Get state dict before compile is allowed.
        init_state_dict = linear_t_g.state_dict()

        of_graph_out = linear_t_g(x)
        iter0_state_dict = linear_t_g.state_dict()

        # Load the model and check
        if call_cnt == 1:
            # Check additional variable state initialized in job has been loaded.
            # TrainStep's placement is only on rank 0, so it needs to be broadcast to all ranks before check.
            cur_train_step = (
                iter0_state_dict["System-Train-TrainStep"]
                .to_global(placement=model_tensor_placement, sbp=B)
                .to_local()
                .numpy()[0]
            )
            test_case.assertEqual(3, cur_train_step)
            test_case.assertTrue(
                cur_train_step
                == last_state_dict["System-Train-TrainStep"]
                .to_global(placement=model_tensor_placement, sbp=B)
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

        # Save the model
        if call_cnt == 0:
            # Transfer the state dict to model_file_placement
            model_file_state_dict = flow.utils.global_view.to_global(
                iter1_state_dict, placement=model_file_placement, sbp=B
            )

            # Get the local component and save it on model_file_placement's rank(s)
            if cur_rank in model_file_placement.ranks:
                iter1_local_dict = flow.utils.global_view.to_local(
                    model_file_state_dict
                )
                flow.save(iter1_local_dict, state_dict_file)

            of_graph_out = linear_t_g(x)
            of_graph_out.numpy()
            iter2_state_dict = linear_t_g.state_dict()
            return iter2_state_dict

    rank_id = flow.env.get_rank()
    with tempfile.NamedTemporaryFile(
        prefix="graph_save_load_global_" + str(rank_id)
    ) as f:
        iter2_state_dict = train_with_graph(0, f.name)
        train_with_graph(1, f.name, iter2_state_dict)


def _test_graph_save_load_global_split_2(
    test_case, model_tensor_placement, model_file_placement
):
    """Pipeline parallelism on 2 ranks.
    """
    P0 = flow.placement(model_tensor_placement.type, ranks=[0])
    P1 = flow.placement(model_tensor_placement.type, ranks=[1])
    BROADCAST = flow.sbp.broadcast

    def get_sbp(state_dict, tensor):
        if tensor is state_dict["System-Train-TrainStep"]:
            return BROADCAST
        if tensor is state_dict["module_pipeline"]["m_stage1.linear.weight"]:
            return flow.sbp.split(1)
        if tensor is state_dict["module_pipeline"]["m_stage1.linear.bias"]:
            return BROADCAST
        return flow.sbp.split(0)

    class Stage0Module(flow.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = flow.nn.Linear(16, 8)
            self.relu = flow.nn.ReLU()

        def forward(self, x):
            return self.relu(self.linear(x))

    class Stage1Module(flow.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = flow.nn.Linear(8, 1)

        def forward(self, x):
            return self.linear(x)

    class PipelineModule(flow.nn.Module):
        def __init__(self):
            super().__init__()
            self.m_stage0 = Stage0Module()
            self.m_stage1 = Stage1Module()

            self.m_stage0.to_global(placement=P0, sbp=BROADCAST)
            self.m_stage1.to_global(placement=P1, sbp=BROADCAST)

        def forward(self, x):
            out_stage0 = self.m_stage0(x)
            in_stage1 = out_stage0.to_global(placement=P1, sbp=BROADCAST)
            out_stage1 = self.m_stage1(in_stage1)
            return out_stage1

    class PipelineGraph(flow.nn.Graph):
        def __init__(self, module_pipleine):
            super().__init__()
            self.module_pipeline = module_pipleine
            self.module_pipeline.m_stage0.to(GraphModule).set_stage(0, P0)
            self.module_pipeline.m_stage1.to(GraphModule).set_stage(1, P1)
            self.config.set_gradient_accumulation_steps(2)
            self.add_optimizer(
                flow.optim.SGD(self.module_pipeline.parameters(), lr=0.001)
            )

        def build(self, x):
            out = self.module_pipeline(x)
            out = out.sum()
            out.backward()
            return out

    def train_with_graph(call_cnt=0, state_dict_file=None, last_state_dict=None):
        # A fixed input with shape [2, 16]
        x = flow.tensor(
            [
                [
                    0.4286,
                    0.7402,
                    0.4161,
                    0.6103,
                    0.7394,
                    1.1330,
                    -0.2311,
                    -0.1013,
                    0.8537,
                    0.9757,
                    -0.9842,
                    0.3839,
                    -0.5551,
                    -0.8832,
                    0.7820,
                    0.7421,
                ],
                [
                    -0.1581,
                    -1.0319,
                    1.8430,
                    0.3576,
                    0.7288,
                    -0.6912,
                    0.9966,
                    1.0840,
                    -1.1760,
                    1.5683,
                    -0.2098,
                    -1.6439,
                    -2.7049,
                    0.1949,
                    1.6377,
                    0.0745,
                ],
            ],
            dtype=oneflow.float32,
            placement=P0,
            sbp=BROADCAST,
        )

        module_pipleine = PipelineModule()
        graph_model = PipelineGraph(module_pipleine)
        cur_rank = flow.env.get_rank()

        if call_cnt == 1:
            if cur_rank in model_file_placement.ranks:
                local_state_dict = flow.load(state_dict_file)
            else:
                local_state_dict = None

            # test sbp_for_special_keys
            global_state_dict = flow.utils.global_view.to_global(
                local_state_dict, placement=model_file_placement, sbp=get_sbp,
            )
            graph_model.load_state_dict(global_state_dict)

            if cur_rank == 0:
                test_case.assertTrue(
                    np.array_equal(
                        global_state_dict["module_pipeline"]["m_stage0.linear.weight"]
                        .to_local()
                        .numpy(),
                        module_pipleine.m_stage0.linear.weight.to_local().numpy()[
                            :4
                        ],  # The first half of shape (8, 16)
                    )
                )
                test_case.assertTrue(
                    np.array_equal(
                        global_state_dict["module_pipeline"]["m_stage0.linear.bias"]
                        .to_local()
                        .numpy(),
                        module_pipleine.m_stage0.linear.bias.to_local().numpy()[
                            :4
                        ],  # The first half of shape (8,)
                    )
                )
            if cur_rank == 1:
                test_case.assertTrue(
                    np.array_equal(
                        global_state_dict["module_pipeline"]["m_stage1.linear.weight"]
                        .to_local()
                        .numpy(),
                        module_pipleine.m_stage1.linear.weight.to_local().numpy()[
                            :, 4:
                        ],  # The second half of shape (1, 8)
                    )
                )
                test_case.assertTrue(
                    np.array_equal(
                        global_state_dict["module_pipeline"]["m_stage1.linear.bias"]
                        .to_local()
                        .numpy(),
                        module_pipleine.m_stage1.linear.bias.to_local().numpy(),
                    )
                )

        graph_model(x)
        iter0_state_dict = graph_model.state_dict()

        if call_cnt == 1:
            # TrainStep
            cur_train_step = (
                iter0_state_dict["System-Train-TrainStep"]
                .to_global(placement=model_tensor_placement, sbp=BROADCAST)
                .to_local()
                .numpy()[0]
            )
            test_case.assertEqual(3, cur_train_step)
            test_case.assertTrue(
                cur_train_step
                == last_state_dict["System-Train-TrainStep"]
                .to_global(placement=model_tensor_placement, sbp=BROADCAST)
                .to_local()
                .numpy()[0]
            )

            # Weight & bias
            test_case.assertTrue(
                np.array_equal(
                    iter0_state_dict["module_pipeline"]["m_stage0.linear.weight"]
                    .to_local()
                    .numpy(),
                    last_state_dict["module_pipeline"]["m_stage0.linear.weight"]
                    .to_local()
                    .numpy(),
                )
            )
            test_case.assertTrue(
                np.array_equal(
                    iter0_state_dict["module_pipeline"]["m_stage0.linear.bias"]
                    .to_local()
                    .numpy(),
                    last_state_dict["module_pipeline"]["m_stage0.linear.bias"]
                    .to_local()
                    .numpy(),
                )
            )
            test_case.assertTrue(
                np.array_equal(
                    iter0_state_dict["module_pipeline"]["m_stage1.linear.weight"]
                    .to_local()
                    .numpy(),
                    last_state_dict["module_pipeline"]["m_stage1.linear.weight"]
                    .to_local()
                    .numpy(),
                )
            )
            test_case.assertTrue(
                np.array_equal(
                    iter0_state_dict["module_pipeline"]["m_stage1.linear.bias"]
                    .to_local()
                    .numpy(),
                    last_state_dict["module_pipeline"]["m_stage1.linear.bias"]
                    .to_local()
                    .numpy(),
                )
            )

        graph_model(x)
        iter1_state_dict = graph_model.state_dict()

        if call_cnt == 0:
            model_file_state_dict = flow.utils.global_view.to_global(
                iter1_state_dict, placement=model_file_placement, sbp=get_sbp,
            )
            if flow.env.get_rank() in model_file_placement.ranks:
                flow.save(
                    flow.utils.global_view.to_local(model_file_state_dict),
                    state_dict_file,
                )

            graph_model(x)
            iter2_state_dict = graph_model.state_dict()
            return iter2_state_dict

    rank_id = flow.env.get_rank()
    with tempfile.NamedTemporaryFile(
        prefix="graph_save_load_global_" + str(rank_id)
    ) as f:
        iter2_state_dict = train_with_graph(0, f.name)
        train_with_graph(1, f.name, iter2_state_dict)


def _test_graph_save_load_global_split_4(
    test_case, model_tensor_placement, model_file_placement
):
    """Pipeline parallelism on 4 ranks.
    """
    P0 = flow.placement(model_tensor_placement.type, ranks=[0])
    P1 = flow.placement(model_tensor_placement.type, ranks=[1])
    P2 = flow.placement(model_tensor_placement.type, ranks=[2])
    P3 = flow.placement(model_tensor_placement.type, ranks=[3])
    BROADCAST = flow.sbp.broadcast

    def get_sbp(state_dict, tensor):
        if tensor is state_dict["System-Train-TrainStep"]:
            return BROADCAST
        if tensor is state_dict["module_pipeline"]["m_stage3.linear.weight"]:
            return flow.sbp.split(1)
        if tensor is state_dict["module_pipeline"]["m_stage3.linear.bias"]:
            return BROADCAST
        return flow.sbp.split(0)

    class Stage0Module(flow.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = flow.nn.Linear(16, 8)
            self.relu = flow.nn.ReLU()

        def forward(self, x):
            return self.relu(self.linear(x))

    class Stage1Module(flow.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = flow.nn.Linear(8, 4)
            self.relu = flow.nn.ReLU()

        def forward(self, x):
            return self.relu(self.linear(x))

    class Stage2Module(flow.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = flow.nn.Linear(4, 2)
            self.relu = flow.nn.ReLU()

        def forward(self, x):
            return self.relu(self.linear(x))

    class Stage3Module(flow.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = flow.nn.Linear(2, 1)

        def forward(self, x):
            return self.linear(x)

    class PipelineModule(flow.nn.Module):
        def __init__(self):
            super().__init__()
            self.m_stage0 = Stage0Module()
            self.m_stage1 = Stage1Module()
            self.m_stage2 = Stage2Module()
            self.m_stage3 = Stage3Module()

            self.m_stage0.to_global(placement=P0, sbp=BROADCAST)
            self.m_stage1.to_global(placement=P1, sbp=BROADCAST)
            self.m_stage2.to_global(placement=P2, sbp=BROADCAST)
            self.m_stage3.to_global(placement=P3, sbp=BROADCAST)

        def forward(self, x):
            out_stage0 = self.m_stage0(x)

            in_stage1 = out_stage0.to_global(placement=P1, sbp=BROADCAST)
            out_stage1 = self.m_stage1(in_stage1)

            in_stage2 = out_stage1.to_global(placement=P2, sbp=BROADCAST)
            out_stage2 = self.m_stage2(in_stage2)

            in_stage3 = out_stage2.to_global(placement=P3, sbp=BROADCAST)
            out_stage3 = self.m_stage3(in_stage3)

            return out_stage3

    class PipelineGraph(flow.nn.Graph):
        def __init__(self, module_pipleine):
            super().__init__()
            self.module_pipeline = module_pipleine
            self.module_pipeline.m_stage0.to(GraphModule).set_stage(0, P0)
            self.module_pipeline.m_stage1.to(GraphModule).set_stage(1, P1)
            self.module_pipeline.m_stage2.to(GraphModule).set_stage(2, P2)
            self.module_pipeline.m_stage3.to(GraphModule).set_stage(3, P3)
            self.config.set_gradient_accumulation_steps(2)
            self.add_optimizer(
                flow.optim.SGD(self.module_pipeline.parameters(), lr=0.001)
            )

        def build(self, x):
            out = self.module_pipeline(x)
            out = out.sum()
            out.backward()
            return out

    def train_with_graph(call_cnt=0, state_dict_file=None, last_state_dict=None):
        # A fixed input with shape [2, 16]
        x = flow.tensor(
            [
                [
                    0.4286,
                    0.7402,
                    0.4161,
                    0.6103,
                    0.7394,
                    1.1330,
                    -0.2311,
                    -0.1013,
                    0.8537,
                    0.9757,
                    -0.9842,
                    0.3839,
                    -0.5551,
                    -0.8832,
                    0.7820,
                    0.7421,
                ],
                [
                    -0.1581,
                    -1.0319,
                    1.8430,
                    0.3576,
                    0.7288,
                    -0.6912,
                    0.9966,
                    1.0840,
                    -1.1760,
                    1.5683,
                    -0.2098,
                    -1.6439,
                    -2.7049,
                    0.1949,
                    1.6377,
                    0.0745,
                ],
            ],
            dtype=flow.float32,
            placement=P0,
            sbp=BROADCAST,
        )

        module_pipleine = PipelineModule()
        graph_model = PipelineGraph(module_pipleine)
        cur_rank = flow.env.get_rank()

        if call_cnt == 1:
            if cur_rank in model_file_placement.ranks:
                local_state_dict = flow.load(state_dict_file)
            else:
                local_state_dict = None

            # test sbp_for_special_keys
            global_state_dict = flow.utils.global_view.to_global(
                local_state_dict, placement=model_file_placement, sbp=get_sbp,
            )
            graph_model.load_state_dict(global_state_dict)

            if cur_rank == 0:
                test_case.assertTrue(
                    np.array_equal(
                        global_state_dict["module_pipeline"]["m_stage0.linear.weight"]
                        .to_local()
                        .numpy(),
                        module_pipleine.m_stage0.linear.weight.to_local().numpy()[
                            :4
                        ],  # The first half of shape (8, 16)
                    )
                )
                test_case.assertTrue(
                    np.array_equal(
                        global_state_dict["module_pipeline"]["m_stage0.linear.bias"]
                        .to_local()
                        .numpy(),
                        module_pipleine.m_stage0.linear.bias.to_local().numpy()[
                            :4
                        ],  # The first half of shape (8,)
                    )
                )
            if cur_rank == 1:
                test_case.assertTrue(
                    np.array_equal(
                        global_state_dict["module_pipeline"]["m_stage1.linear.weight"]
                        .to_local()
                        .numpy(),
                        module_pipleine.m_stage1.linear.weight.to_local().numpy()[
                            2:, :
                        ],  # The second half of shape (4, 8)
                    )
                )
                test_case.assertTrue(
                    np.array_equal(
                        global_state_dict["module_pipeline"]["m_stage1.linear.bias"]
                        .to_local()
                        .numpy(),
                        module_pipleine.m_stage1.linear.bias.to_local().numpy()[
                            2:
                        ],  # The second half if shape (4,)
                    )
                )

        graph_model(x)
        iter0_state_dict = graph_model.state_dict()

        if call_cnt == 1:
            # TrainStep
            cur_train_step = (
                iter0_state_dict["System-Train-TrainStep"]
                .to_global(placement=model_tensor_placement, sbp=BROADCAST)
                .to_local()
                .numpy()[0]
            )
            test_case.assertEqual(3, cur_train_step)
            test_case.assertTrue(
                cur_train_step
                == last_state_dict["System-Train-TrainStep"]
                .to_global(placement=model_tensor_placement, sbp=BROADCAST)
                .to_local()
                .numpy()[0]
            )

            # Weight & bias
            test_case.assertTrue(
                np.array_equal(
                    iter0_state_dict["module_pipeline"]["m_stage0.linear.weight"]
                    .to_local()
                    .numpy(),
                    last_state_dict["module_pipeline"]["m_stage0.linear.weight"]
                    .to_local()
                    .numpy(),
                )
            )
            test_case.assertTrue(
                np.array_equal(
                    iter0_state_dict["module_pipeline"]["m_stage0.linear.bias"]
                    .to_local()
                    .numpy(),
                    last_state_dict["module_pipeline"]["m_stage0.linear.bias"]
                    .to_local()
                    .numpy(),
                )
            )
            test_case.assertTrue(
                np.array_equal(
                    iter0_state_dict["module_pipeline"]["m_stage1.linear.weight"]
                    .to_local()
                    .numpy(),
                    last_state_dict["module_pipeline"]["m_stage1.linear.weight"]
                    .to_local()
                    .numpy(),
                )
            )
            test_case.assertTrue(
                np.array_equal(
                    iter0_state_dict["module_pipeline"]["m_stage1.linear.bias"]
                    .to_local()
                    .numpy(),
                    last_state_dict["module_pipeline"]["m_stage1.linear.bias"]
                    .to_local()
                    .numpy(),
                )
            )

            test_case.assertTrue(
                np.array_equal(
                    iter0_state_dict["module_pipeline"]["m_stage2.linear.weight"]
                    .to_local()
                    .numpy(),
                    last_state_dict["module_pipeline"]["m_stage2.linear.weight"]
                    .to_local()
                    .numpy(),
                )
            )
            test_case.assertTrue(
                np.array_equal(
                    iter0_state_dict["module_pipeline"]["m_stage2.linear.bias"]
                    .to_local()
                    .numpy(),
                    last_state_dict["module_pipeline"]["m_stage2.linear.bias"]
                    .to_local()
                    .numpy(),
                )
            )

            test_case.assertTrue(
                np.array_equal(
                    iter0_state_dict["module_pipeline"]["m_stage3.linear.weight"]
                    .to_local()
                    .numpy(),
                    last_state_dict["module_pipeline"]["m_stage3.linear.weight"]
                    .to_local()
                    .numpy(),
                )
            )
            test_case.assertTrue(
                np.array_equal(
                    iter0_state_dict["module_pipeline"]["m_stage3.linear.bias"]
                    .to_local()
                    .numpy(),
                    last_state_dict["module_pipeline"]["m_stage3.linear.bias"]
                    .to_local()
                    .numpy(),
                )
            )

        graph_model(x)
        iter1_state_dict = graph_model.state_dict()

        if call_cnt == 0:
            model_file_state_dict = flow.utils.global_view.to_global(
                iter1_state_dict, placement=model_file_placement, sbp=get_sbp,
            )
            if flow.env.get_rank() in model_file_placement.ranks:
                flow.save(
                    flow.utils.global_view.to_local(model_file_state_dict),
                    state_dict_file,
                )

            graph_model(x)
            iter2_state_dict = graph_model.state_dict()
            return iter2_state_dict

    rank_id = flow.env.get_rank()
    with tempfile.NamedTemporaryFile(
        prefix="graph_save_load_global_" + str(rank_id)
    ) as f:
        iter2_state_dict = train_with_graph(0, f.name)
        train_with_graph(1, f.name, iter2_state_dict)


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
@flow.unittest.skip_unless_1n2d()
class TestGraphSaveLoadGlobal2d(oneflow.unittest.TestCase):
    def test_linear_graph_save_load_gpu_1_broadcast(test_case):
        _test_linear_graph_save_load_global_broadcast(
            test_case,
            model_tensor_placement=flow.placement("cuda", ranks=[0, 1]),
            model_file_placement=flow.placement("cpu", ranks=[0]),
        )

    def test_linear_graph_save_load_cpu_1_broadcast(test_case):
        _test_linear_graph_save_load_global_broadcast(
            test_case,
            model_tensor_placement=flow.placement("cpu", ranks=[0, 1]),
            model_file_placement=flow.placement("cpu", ranks=[0]),
        )

    def test_graph_save_load_gpu_2_split(test_case):
        _test_graph_save_load_global_split_2(
            test_case,
            model_tensor_placement=flow.placement("cuda", ranks=[0, 1]),
            model_file_placement=flow.placement("cpu", ranks=[0, 1]),
        )

    def test_graph_save_load_cpu_2_split(test_case):
        _test_graph_save_load_global_split_2(
            test_case,
            model_tensor_placement=flow.placement("cpu", ranks=[0, 1]),
            model_file_placement=flow.placement("cpu", ranks=[0, 1]),
        )


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
@flow.unittest.skip_unless_1n4d()
class TestGraphSaveLoadGlobal4d(oneflow.unittest.TestCase):
    def test_graph_save_load_gpu_2_split_2_none(test_case):
        _test_graph_save_load_global_split_4(
            test_case,
            model_tensor_placement=flow.placement("cuda", ranks=[0, 1, 2, 3]),
            model_file_placement=flow.placement("cpu", ranks=[0, 1]),
        )

    def test_graph_save_load_cpu_2_split_2_none(test_case):
        _test_graph_save_load_global_split_4(
            test_case,
            model_tensor_placement=flow.placement("cpu", ranks=[0, 1, 2, 3]),
            model_file_placement=flow.placement("cpu", ranks=[0, 1]),
        )


if __name__ == "__main__":
    unittest.main()
