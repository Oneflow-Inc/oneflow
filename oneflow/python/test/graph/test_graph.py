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

import oneflow
import oneflow.experimental as flow
import oneflow.python.framework.graph_build_util as graph_build_util


class SubModule(flow.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = flow.nn.Conv2d(1, 1, 5)
        self.relu = flow.nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        return x


class CustomModule(flow.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = SubModule()
        self.fc1 = flow.nn.Linear(36, 4)
        self.register_buffer(
            "dummy_buff", flow.Tensor(1, 4),
        )

    def forward(self, x):
        x = self.layer(x)
        x = oneflow.F.flatten(x, 1)
        x = self.fc1(x) + self.dummy_buff
        return x


@flow.unittest.skip_unless_1n1d()
class TestGraph(flow.unittest.TestCase):
    def test_add_nested_module(test_case):
        x = flow.Tensor(1, 1, 10, 10)
        flow.nn.init.uniform_(x, a=-1.0, b=1.0)

        # Module init and call
        m = CustomModule()
        y = m(x)

        class CustomGraph(flow.nn.Graph):
            def __init__(self):
                super().__init__()
                self.m = m

            def build(self, x):
                return self.m(x)

        # Graph init
        g = CustomGraph()
        # check _c_nn_graph init
        test_case.assertEqual(g.name, g._c_nn_graph.name)
        # g.m is Block
        test_case.assertTrue(isinstance(g.m, flow.nn.graph.Block))
        test_case.assertEqual(g.m.type, "MODULE")
        # g.m.name is "m"
        test_case.assertEqual(g.m.name, "m")
        # g.m.dummy_buff is Block
        test_case.assertTrue(isinstance(g.m.dummy_buff, flow.nn.graph.Block))
        test_case.assertEqual(g.m.dummy_buff.type, "BUFFER")

        # conv1 is Block
        test_case.assertTrue(isinstance(g.m.layer.conv1, flow.nn.graph.Block))
        # conv1.name is "conv1"
        test_case.assertEqual(g.m.layer.conv1.name, "conv1")
        # conv1.name_prefix is "m.layer."
        test_case.assertEqual(g.m.layer.conv1.name_prefix, "m.layer.")
        # conv1.weight is Block
        test_case.assertTrue(isinstance(g.m.layer.conv1.weight, flow.nn.graph.Block))
        test_case.assertEqual(g.m.layer.conv1.weight.type, "PARAMETER")
        # conv1.weight is Tensor, Graph.build(...) need weight to be Tensor
        g.m.layer.conv1._is_executing_forward = True
        test_case.assertTrue(isinstance(g.m.layer.conv1.weight, flow.Tensor))
        g.m.layer.conv1._is_executing_forward = False
        # conv1.kernel_size is original data in original module
        test_case.assertEqual(g.m.layer.conv1.kernel_size, (5, 5))

        # Graph build
        z = g.build(x)
        # g got the same result as m
        test_case.assertTrue(np.array_equal(y.numpy(), z.numpy()))

    def test_graph_config(test_case):
        class CustomGraph(flow.nn.Graph):
            def __init__(self):
                super().__init__()
                self.m = CustomModule()
                self.config.enable_auto_mixed_precision(True)

            def build(self, x):
                x = self.m(x)
                return x

        g = CustomGraph()

        # check default training is True
        test_case.assertEqual(g.config.training, False)

        # set graph config
        g.config.enable_fuse_add_to_output(True)
        g.config.enable_fuse_add_to_output(False)

        for s in g._state():
            print("g state: ", repr(s))

        # print repr of nn.Graph
        print(repr(g))

    def test_graph_name(test_case):
        class ACustomGraph(flow.nn.Graph):
            def __init__(self):
                super().__init__()

            def build(self, x):
                return x

        class BCustomGraph(flow.nn.Graph):
            def __init__(self):
                super().__init__()

            def build(self, x):
                return x

        class CBCustomGraph(BCustomGraph):
            def __init__(self):
                super().__init__()

        def create_graph(cnt):
            a = ACustomGraph()
            test_case.assertEqual(a.name, "ACustomGraph_" + str(cnt))
            b = BCustomGraph()
            test_case.assertEqual(b.name, "BCustomGraph_" + str(cnt))
            cb = CBCustomGraph()
            test_case.assertEqual(cb.name, "CBCustomGraph_" + str(cnt))

        flow.nn.Graph._child_init_cnt.clear()
        for i in range(0, 3):
            create_graph(i)
        flow.nn.Graph._child_init_cnt.clear()
        for i in range(0, 3):
            create_graph(i)

    def test_graph_build_ctx(test_case):

        # check lazy_mode
        test_case.assertEqual(graph_build_util.lazy_mode.is_enabled(), False)
        with graph_build_util.lazy_mode.gard(True):
            test_case.assertEqual(graph_build_util.lazy_mode.is_enabled(), True)
            with graph_build_util.lazy_mode.gard(False):
                test_case.assertEqual(graph_build_util.lazy_mode.is_enabled(), False)
            test_case.assertEqual(graph_build_util.lazy_mode.is_enabled(), True)
        test_case.assertEqual(graph_build_util.lazy_mode.is_enabled(), False)

        class CustomGraph(flow.nn.Graph):
            def __init__(self):
                super().__init__()
                self.config.enable_auto_mixed_precision(True)

            def build(self):
                # check lazy mode in nn.Graph._compile
                test_case.assertEqual(graph_build_util.lazy_mode.is_enabled(), True)

                # check session type
                import oneflow.python.framework.session_context as session_ctx
                from oneflow.python.framework.multi_client_session import (
                    MultiClientSession,
                )

                session = session_ctx.GetDefaultSession()
                test_case.assertEqual(type(session), MultiClientSession)

                # check scope
                import oneflow.python.framework.scope_util as scope_util

                scope = oneflow.current_scope()
                scope_proto = graph_build_util.scope_to_proto(scope)
                test_case.assertEqual(session.id, scope_proto.session_id)

                # check job_build_and_infer_ctx
                test_case.assertEqual(
                    oneflow._oneflow_internal.JobBuildAndInferCtx_GetCurrentJobName(),
                    self.name,
                )

        test_case.assertTrue(oneflow._oneflow_internal.IsMultiClient())
        g = CustomGraph()
        test_case.assertEqual(graph_build_util.lazy_mode.is_enabled(), False)
        g._compile()
        print("graph proto", g._graph_proto)
        test_case.assertEqual(graph_build_util.lazy_mode.is_enabled(), False)

    def test_block_scope(test_case):
        class SubModule0(flow.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = flow.nn.Conv2d(1, 1, 5)

            def forward(self):
                scope = oneflow.current_scope()
                scope_proto = graph_build_util.scope_to_proto(scope)

                # check scope activation checkpointing
                ck_bool = scope_proto.attr_name2attr_value["checkpointing"].at_bool
                test_case.assertEqual(ck_bool, True)
                # check scope stage id
                stage_int = scope_proto.attr_name2attr_value[
                    "pipeline_stage_id_hint"
                ].at_int64
                test_case.assertEqual(stage_int, 0)

                # weight is not get in conv1's forward, so it will return a Block
                x = self.conv1.weight
                test_case.assertEqual(type(x), flow.nn.graph.Block)
                return x

        class SubModule1(flow.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = flow.nn.Linear(36, 4)
                self.register_buffer(
                    "dummy_buff", flow.Tensor(1, 4),
                )

            def forward(self):
                scope = oneflow.current_scope()
                scope_proto = graph_build_util.scope_to_proto(scope)

                # check scope symbol id
                test_case.assertEqual(
                    scope_proto.parent_scope_symbol_id, self.prev_scope.symbol_id
                )

                # check scope activation checkpointing
                ck_bool = scope_proto.attr_name2attr_value["checkpointing"]
                test_case.assertEqual(ck_bool.WhichOneof("value"), None)
                # check scope stage id
                stage_int = scope_proto.attr_name2attr_value[
                    "pipeline_stage_id_hint"
                ].at_int64
                test_case.assertEqual(stage_int, 1)

                name = self.name_prefix + self.name
                prefixes = []
                for prefix in scope_proto.scope_op_name_prefixes:
                    prefixes.append(prefix)
                name_in_scope = ".".join(prefixes)
                test_case.assertEqual(name, name_in_scope)

                x = self.dummy_buff
                dummy_buff_scope_proto = graph_build_util.scope_to_proto(
                    self._buffers["dummy_buff"].scope
                )
                test_case.assertEqual(
                    dummy_buff_scope_proto.parent_scope_symbol_id, scope.symbol_id
                )
                return x

        class CustomModule1(flow.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer0 = SubModule0()
                self.layer1 = SubModule1()

            def forward(self):
                x = self.layer0()
                y = self.layer1()
                return x, y

        m = CustomModule1()

        class CustomGraph1(flow.nn.Graph):
            def __init__(self):
                super().__init__()
                self.m = m
                # config scope
                self.m.layer0.config.stage_id = 0
                self.m.layer0.config.activation_checkpointing = True
                self.m.layer1.config.stage_id = 1

            def build(self):
                return self.m()

        g = CustomGraph1()
        x = flow.Tensor(1, 1, 10, 10)
        flow.nn.init.uniform_(x, a=-1.0, b=1.0)
        z = g._compile()


if __name__ == "__main__":
    unittest.main()
