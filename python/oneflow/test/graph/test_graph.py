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
from threading import Thread

import numpy as np

import oneflow
import oneflow as flow
from oneflow.nn.graph import GraphModule, GraphTensor
import oneflow.framework.graph_build_util as graph_build_util
import oneflow.framework.scope_util as scope_util
import oneflow.unittest


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
        self.register_buffer("dummy_buff", flow.Tensor(1, 4))

    def forward(self, x):
        x = self.layer(x)
        x = oneflow._C.flatten(x, 1)
        x = self.fc1(x) + self.dummy_buff
        return x


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
@flow.unittest.skip_unless_1n1d()
class TestGraph(flow.unittest.TestCase):
    def test_add_nested_module(test_case):
        x = flow.Tensor(1, 1, 10, 10)
        flow.nn.init.uniform_(x, a=-1.0, b=1.0)
        m = CustomModule()
        y = m(x)

        class CustomGraphNestedModule(flow.nn.Graph):
            def __init__(self):
                super().__init__()
                self.m = m

            def build(self, x):
                return self.m(x)

        g = CustomGraphNestedModule()
        test_case.assertTrue(isinstance(g.m, flow.nn.graph.Proxy))
        test_case.assertEqual(g.m.to(GraphModule).type, "MODULE")
        test_case.assertEqual(g.m.to(GraphModule).name, "m")
        test_case.assertTrue(isinstance(g.m.dummy_buff, flow.nn.graph.Proxy))
        test_case.assertEqual(g.m.dummy_buff.to(GraphTensor).type, "BUFFER")
        test_case.assertTrue(isinstance(g.m.layer.conv1, flow.nn.graph.Proxy))
        test_case.assertEqual(g.m.layer.conv1.to(GraphModule).name, "conv1")
        test_case.assertEqual(g.m.layer.conv1.to(GraphModule).name_prefix, "m.layer.")
        test_case.assertTrue(isinstance(g.m.layer.conv1.weight, flow.nn.graph.Proxy))
        test_case.assertEqual(g.m.layer.conv1.weight.to(GraphTensor).type, "PARAMETER")
        g.m.layer.conv1.to(GraphModule)._is_executing_forward = True
        test_case.assertTrue(isinstance(g.m.layer.conv1.weight, flow.Tensor))
        g.m.layer.conv1.to(GraphModule)._is_executing_forward = False
        test_case.assertEqual(g.m.layer.conv1.kernel_size, (5, 5))
        z = g.build(x)
        test_case.assertTrue(np.array_equal(y.numpy(), z.numpy()))

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
        test_case.assertEqual(graph_build_util.lazy_mode.is_enabled(), False)
        with graph_build_util.lazy_mode.guard(True):
            test_case.assertEqual(graph_build_util.lazy_mode.is_enabled(), True)
            with graph_build_util.lazy_mode.guard(False):
                test_case.assertEqual(graph_build_util.lazy_mode.is_enabled(), False)
            test_case.assertEqual(graph_build_util.lazy_mode.is_enabled(), True)
        test_case.assertEqual(graph_build_util.lazy_mode.is_enabled(), False)

        class CustomGraphGraphBuildCtx(flow.nn.Graph):
            def __init__(self):
                super().__init__()

            def build(self, x):
                test_case.assertEqual(graph_build_util.lazy_mode.is_enabled(), True)
                import oneflow.framework.session_context as session_ctx
                from oneflow.framework.multi_client_session import MultiClientSession

                session = session_ctx.GetDefaultSession()
                test_case.assertEqual(type(session), MultiClientSession)
                import oneflow.framework.scope_util as scope_util

                scope = scope_util.current_scope()
                scope_proto = graph_build_util.scope_to_proto(scope)
                test_case.assertEqual(session.id, scope_proto.session_id)
                test_case.assertEqual(
                    oneflow._oneflow_internal.JobBuildAndInferCtx_GetCurrentJobName(),
                    self.name,
                )
                return x

        g = CustomGraphGraphBuildCtx()
        test_case.assertEqual(graph_build_util.lazy_mode.is_enabled(), False)
        data = np.array([2.0, 1.0, 0.0, -1.0, -2.0])
        x = flow.tensor(data, dtype=flow.float32)
        g._compile(x)
        print("graph proto", g._graph_proto)
        test_case.assertEqual(graph_build_util.lazy_mode.is_enabled(), False)

    def test_block_scope(test_case):
        class SubModule0(flow.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = flow.nn.Conv2d(1, 1, 5)

            def forward(self, x):
                scope = scope_util.current_scope()
                scope_proto = graph_build_util.scope_to_proto(scope)
                ck_bool = scope_proto.attr_name2attr_value["checkpointing"].at_bool
                test_case.assertEqual(ck_bool, True)
                stage_int = scope_proto.attr_name2attr_value[
                    "pipeline_stage_id_hint"
                ].at_int64
                test_case.assertEqual(stage_int, 0)
                out = self.conv1(x)
                weight = self.conv1.weight
                test_case.assertTrue(weight.is_lazy)
                return out

        class SubModule1(flow.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = flow.nn.Linear(36, 4, False)
                self.register_buffer("dummy_buff", flow.Tensor(1, 4))

            def forward(self, x):
                scope = scope_util.current_scope()
                scope_proto = graph_build_util.scope_to_proto(scope)
                test_case.assertEqual(
                    scope_proto.parent_scope_symbol_id,
                    self.to(flow.nn.graph.GraphModule).prev_scope.symbol_id,
                )
                ck_bool = scope_proto.attr_name2attr_value["checkpointing"]
                test_case.assertEqual(ck_bool.WhichOneof("value"), None)
                stage_int = scope_proto.attr_name2attr_value[
                    "pipeline_stage_id_hint"
                ].at_int64
                test_case.assertEqual(stage_int, 1)
                name = (
                    self.to(flow.nn.graph.GraphModule).name_prefix
                    + self.to(flow.nn.graph.GraphModule).name
                )
                prefixes = []
                for prefix in scope_proto.scope_op_name_prefixes:
                    prefixes.append(prefix)
                name_in_scope = ".".join(prefixes)
                test_case.assertEqual(name, name_in_scope)
                b = self.dummy_buff
                dummy_buff_scope_proto = graph_build_util.scope_to_proto(
                    self._buffers["dummy_buff"].to(flow.nn.graph.GraphTensor).scope
                )
                test_case.assertEqual(
                    dummy_buff_scope_proto.parent_scope_symbol_id, scope.symbol_id
                )
                x = self.fc1(x)
                return x + b

        class CustomModule1(flow.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer0 = SubModule0()
                self.layer1 = SubModule1()

            def forward(self, x, y):
                print("x0: ", x.shape)
                x = self.layer0(x)
                print("x1: ", x.shape)
                print("y0: ", y.shape)
                y = self.layer1(y)
                print("y1: ", y.shape)
                return (x, y)

        m = CustomModule1()

        class CustomGraphBlockScope(flow.nn.Graph):
            def __init__(self):
                super().__init__()
                self.m = m
                self.m.layer0.to(GraphModule).set_stage(stage_id=0)
                self.m.layer0.to(GraphModule).activation_checkpointing = True
                self.m.layer1.to(GraphModule).set_stage(stage_id=1)

            def build(self, x, y):
                return self.m(x, y)

        g = CustomGraphBlockScope()
        print(g)
        x = np.ones((1, 1, 10, 10))
        x = flow.tensor(x, dtype=flow.float32)
        y = np.ones((16, 36))
        y = flow.tensor(y, dtype=flow.float32)
        g._compile(x, y)

    def test_create_optimizer_in_graph(test_case):
        device = "cuda"
        linear = flow.nn.Linear(3, 8)
        linear = linear.to(device)
        flow.nn.init.constant_(linear.weight, 2.068758)
        flow.nn.init.constant_(linear.bias, 0.23)

        class OptCreatedInGraph(flow.nn.Graph):
            def __init__(self):
                super().__init__()
                self.linear = linear
                # creat optimizer in nn.Graph and add parameter from ProxyModule
                self.add_optimizer(
                    flow.optim.SGD(self.linear.parameters(), lr=0.001, momentum=0.9)
                )

            def build(self, x):
                out = self.linear(x)
                out = out.sum()
                out.backward()
                return out

        g = OptCreatedInGraph()
        print(g)

    def test_graph_in_subthread(test_case):
        class TinyGraph(flow.nn.Graph):
            def __init__(self):
                super().__init__()

            def build(self, input):
                return input + 1

        def f():
            tiny_graph = TinyGraph()
            input = flow.randn(1, 4)
            return tiny_graph(input)

        f()

        new_thread = Thread(target=f)

        new_thread.start()
        new_thread.join()


if __name__ == "__main__":
    unittest.main()
