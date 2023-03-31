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
import types
import warnings

import numpy as np

import oneflow as flow
import oneflow.nn as nn
import oneflow.unittest
import oneflow.framework.graph_build_util as graph_build_util
import oneflow.framework.scope_util as scope_util


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
@flow.unittest.skip_unless_1n1d()
class TestGraphBlock(flow.unittest.TestCase):
    def test_module_has_custom_func(test_case):
        class CustomModuleHasFunc(flow.nn.Module):
            def __init__(self):
                super().__init__()
                self.data_mem = 10

            def forward(self, x):
                return self._custom_func(x)

            def _custom_func(self, x):
                test_case.assertEqual(self.data_mem, 10)
                return x

        class CustomGraphHasFunc(flow.nn.Graph):
            def __init__(self):
                super().__init__()
                self.m = CustomModuleHasFunc()

            def build(self, x):
                return self.m(x)

        g = CustomGraphHasFunc()
        x = np.ones((10, 10))
        x = flow.tensor(x, dtype=flow.float32)
        out = g(x)
        test_case.assertTrue(np.array_equal(x.numpy(), out.numpy()))

    def test_module_has_special_attr(test_case):
        class CustomModuleHasSpecialAttr(flow.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = 1
                self.name = "test_name"

            def forward(self, x):
                test_case.assertEqual(self.config, 1)
                test_case.assertEqual(self.name, "test_name")
                test_case.assertEqual(self.to(nn.graph.GraphModule).name, "m")
                return x

        class CustomGraphHasSpecialAttr(flow.nn.Graph):
            def __init__(self):
                super().__init__()
                self.m = CustomModuleHasSpecialAttr()

            def build(self, x):
                return self.m(x)

        g = CustomGraphHasSpecialAttr()
        x = np.ones((10, 10))
        x = flow.tensor(x, dtype=flow.float32)
        out = g(x)
        test_case.assertTrue(np.array_equal(x.numpy(), out.numpy()))

    def test_block_with_parameter(test_case):
        device = "cuda"
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

        class CustomModule(flow.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = linear

            def forward(self, x):
                return self._forward_impl(x)

            def _forward_impl(self, x):
                test_case.assertTrue(isinstance(self.linear, flow.nn.graph.Proxy))
                return self.linear(x)

        class LinearTrainGraph(flow.nn.Graph):
            def __init__(self):
                super().__init__()
                self.m = CustomModule()
                self.add_optimizer(of_sgd)

            def build(self, x):
                out = self.m(x)
                out = out.sum()
                out.backward()
                test_case.assertTrue(self.m.linear.weight.is_lazy)
                return out

        linear_t_g = LinearTrainGraph()

        linear_t_g(x)

    def test_block_get_class_in_forward(test_case):
        device = "cuda"
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

        class CustomModule(flow.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = linear

            def forward(self, x):
                return self._forward_impl(x)

            def _forward_impl(self, x):
                test_case.assertTrue(isinstance(self.linear, flow.nn.Module))
                test_case.assertTrue(isinstance(self.linear, flow.nn.Linear))
                return self.linear(x)

        class LinearTrainGraph(flow.nn.Graph):
            def __init__(self):
                super().__init__()
                self.m = CustomModule()
                test_case.assertTrue(isinstance(self.m.linear, flow.nn.graph.Proxy))
                self.add_optimizer(of_sgd)

            def build(self, x):
                test_case.assertTrue(isinstance(self.m.linear, flow.nn.Module))
                test_case.assertTrue(isinstance(self.m.linear, flow.nn.Linear))
                out = self.m(x)
                out = out.sum()
                out.backward()
                test_case.assertTrue(self.m.linear.weight.is_lazy)
                return out

        linear_t_g = LinearTrainGraph()
        test_case.assertTrue(isinstance(linear_t_g.m.linear, flow.nn.graph.Proxy))

        linear_t_g(x)

    def test_block_with_not_registered_module(test_case):
        device = "cuda"
        linear = flow.nn.Linear(3, 8)
        linear = linear.to(device)
        flow.nn.init.constant_(linear.weight, 2.068758)
        flow.nn.init.constant_(linear.bias, 0.23)

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

        class CustomModule(flow.nn.Module):
            def __init__(self):
                super().__init__()
                self.dict = {"lin": linear}

            def forward(self, x):
                return self._forward_impl(x)

            def _forward_impl(self, x):
                test_case.assertTrue(isinstance(self.dict["lin"], flow.nn.Module))
                return self.dict["lin"](x)

        class LinearTrainGraph(flow.nn.Graph):
            def __init__(self):
                super().__init__()
                self.m = CustomModule()

            def build(self, x):
                out = self.m(x)
                out = out.sum()
                return out

        linear_t_g = LinearTrainGraph()

        with warnings.catch_warnings(record=True) as w:
            # Here will print:
            #     UserWarning: Linear(in_features=3, out_features=8, bias=True) is called in a nn.Graph, but not registered into a nn.Graph.
            linear_t_g(x)

            test_case.assertTrue(len(w) == 1)
            test_case.assertTrue(issubclass(w[-1].category, UserWarning))
            test_case.assertTrue(
                "is called in a nn.Graph, but not registered into a nn.Graph"
                in str(w[-1].message)
            )

    def test_block_with_seq_container(test_case):
        class SubModule0(flow.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = flow.nn.Linear(10, 10, False)

            def forward(self, x):
                if graph_build_util.lazy_mode.is_enabled():
                    scope = scope_util.current_scope()
                    scope_proto = graph_build_util.scope_to_proto(scope)
                    ck_bool = scope_proto.attr_name2attr_value["checkpointing"].at_bool
                    test_case.assertEqual(ck_bool, True)
                out = self.linear(x)
                return out

        list_of_m = [SubModule0() for i in range(3)]

        class SeqModule(flow.nn.Module):
            def __init__(self):
                super().__init__()
                self.linears = flow.nn.Sequential(*list_of_m)

            def forward(self, x):
                x = self.linears(x)
                return x

        class SeqGraph(flow.nn.Graph):
            def __init__(self):
                super().__init__()
                self.linears = flow.nn.Sequential(*list_of_m)
                self.linears.to(nn.graph.GraphModule).activation_checkpointing = True

            def build(self, x):
                x = self.linears(x)
                return x

        seq_m = SeqModule()
        seq_g = SeqGraph()

        input = flow.tensor(np.random.randn(4, 10), dtype=flow.float32)
        output_m = seq_m(input)
        output_g = seq_g(input)

        # print(seq_g)
        test_case.assertTrue(np.array_equal(output_m.numpy(), output_g.numpy()))

    def test_block_with_list_container(test_case):
        class SubModule0(flow.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = flow.nn.Linear(10, 10, False)

            def forward(self, x):
                if graph_build_util.lazy_mode.is_enabled():
                    scope = scope_util.current_scope()
                    scope_proto = graph_build_util.scope_to_proto(scope)
                    ck_bool = scope_proto.attr_name2attr_value["checkpointing"].at_bool
                    test_case.assertEqual(ck_bool, True)
                out = self.linear(x)
                return out

        list_of_m = [SubModule0() for i in range(3)]

        class ModuleListModule(flow.nn.Module):
            def __init__(self):
                super().__init__()
                self.linears = flow.nn.ModuleList(list_of_m)

            def forward(self, x):
                for i, _ in enumerate(self.linears):
                    x = self.linears[i](x)
                return x

        class ModuleListGraph(flow.nn.Graph):
            def __init__(self):
                super().__init__()
                self.linears = flow.nn.ModuleList(list_of_m)
                # NOTE: ModuleList doesn't have config.
                # self.linears.to(GraphModule).activation_checkpointing = True
                for i, _ in enumerate(self.linears):
                    self.linears[i].to(
                        nn.graph.GraphModule
                    ).activation_checkpointing = True

            def build(self, x):
                # ModuleList can act as an iterable, or be indexed using ints
                for i, _ in enumerate(self.linears):
                    x = self.linears[i](x)

                return x

        module_list_m = ModuleListModule()
        module_list_g = ModuleListGraph()

        input = flow.tensor(np.random.randn(4, 10), dtype=flow.float32)
        output_m = module_list_m(input)
        output_g = module_list_g(input)

        # print(module_list_g)
        test_case.assertTrue(np.array_equal(output_m.numpy(), output_g.numpy()))

    def test_module_list_slice(test_case):
        class ModuleListSlice(nn.Module):
            def __init__(self,):
                super().__init__()
                linear1 = nn.Linear(5, 5, bias=False)
                linear2 = nn.Linear(5, 5, bias=False)
                linear3 = nn.Linear(5, 5, bias=False)
                self.modulelist = nn.ModuleList([linear1, linear2, linear3])

            def forward(self, x):
                sliced_m = self.modulelist[1:]
                test_case.assertEqual(len(sliced_m), 2)
                y = sliced_m[1](x)
                return y

        class GraphModuleListSlice(nn.Graph):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def build(self, x):
                return self.m(x)

        in_tensor = flow.randn(5, 5)

        m = ModuleListSlice()
        eager_out = m(in_tensor)

        g = GraphModuleListSlice(m)
        graph_out = g(in_tensor)

        test_case.assertTrue(np.array_equal(eager_out.numpy(), graph_out.numpy()))

    def test_block_with_dict_container(test_case):
        class SubModule0(flow.nn.Module):
            def __init__(self, out):
                super().__init__()
                self.linear = flow.nn.Linear(10, out, False)

            def forward(self, x):
                if graph_build_util.lazy_mode.is_enabled():
                    scope = scope_util.current_scope()
                    scope_proto = graph_build_util.scope_to_proto(scope)
                    ck_bool = scope_proto.attr_name2attr_value["checkpointing"].at_bool
                    test_case.assertEqual(ck_bool, True)
                out = self.linear(x)
                return out

        dict_of_m = {"0": SubModule0(10), "1": SubModule0(6)}

        class ModuleDictModule(flow.nn.Module):
            def __init__(self):
                super().__init__()
                self.linears = flow.nn.ModuleDict(dict_of_m)

            def forward(self, x):
                x = self.linears["0"](x)
                x = self.linears["1"](x)
                return x

        class ModuleDictGraph(flow.nn.Graph):
            def __init__(self):
                super().__init__()
                self.linears = flow.nn.ModuleDict(dict_of_m)

                # NOTE: ModuleDict doesn't have config.
                # self.linears.to(GraphModule).activation_checkpointing = True
                for k, _ in self.linears.items():
                    self.linears[k].to(
                        nn.graph.GraphModule
                    ).activation_checkpointing = True

            def build(self, x):
                # ModuleDict can act as an iterable, or get using key
                x = self.linears["0"](x)
                x = self.linears["1"](x)
                return x

        module_dict_m = ModuleDictModule()
        module_dict_g = ModuleDictGraph()

        input = flow.tensor(np.random.randn(4, 10), dtype=flow.float32)
        output_m = module_dict_m(input)
        output_g = module_dict_g(input)

        # print(module_dict_g)
        test_case.assertTrue(np.array_equal(output_m.numpy(), output_g.numpy()))

    def test_block_with_dict_container_nto1(test_case):
        class SubModule0(flow.nn.Module):
            def __init__(self, out):
                super().__init__()
                self.linear = flow.nn.Linear(10, out, False)

            def forward(self, x):
                if graph_build_util.lazy_mode.is_enabled():
                    scope = scope_util.current_scope()
                    scope_proto = graph_build_util.scope_to_proto(scope)
                    ck_bool = scope_proto.attr_name2attr_value["checkpointing"].at_bool
                    test_case.assertEqual(ck_bool, True)
                out = self.linear(x)
                return out

        sub_m = SubModule0(10)
        dict_of_m = {"0": sub_m, "1": sub_m}

        class ModuleDictModule(flow.nn.Module):
            def __init__(self):
                super().__init__()
                self.linears = flow.nn.ModuleDict(dict_of_m)

            def forward(self, x):
                x = self.linears["0"](x)
                x = self.linears["1"](x)
                return x

        class ModuleDictGraph(flow.nn.Graph):
            def __init__(self):
                super().__init__()
                self.linears = flow.nn.ModuleDict(dict_of_m)

                # NOTE: ModuleDict doesn't have config.
                # self.linears.to(GraphModule).activation_checkpointing = True
                for k, _ in self.linears.items():
                    self.linears[k].to(
                        nn.graph.GraphModule
                    ).activation_checkpointing = True

            def build(self, x):
                # ModuleDict can act as an iterable, or get using key
                x = self.linears["0"](x)
                x = self.linears["1"](x)
                return x

        module_dict_m = ModuleDictModule()
        module_dict_g = ModuleDictGraph()

        input = flow.tensor(np.random.randn(4, 10), dtype=flow.float32)
        output_m = module_dict_m(input)
        output_g = module_dict_g(input)
        print(module_dict_g)

        # print(module_dict_g)
        test_case.assertTrue(np.array_equal(output_m.numpy(), output_g.numpy()))

    def test_block_with_para_list_container(test_case):
        list_of_p = [flow.nn.Parameter(flow.randn(10, 10)) for i in range(2)]

        class ParaListModule(flow.nn.Module):
            def __init__(self):
                super().__init__()
                self.params = flow.nn.ParameterList(list_of_p)

            def forward(self, x):
                for i, _ in enumerate(self.params):
                    x = flow._C.matmul(x, self.params[i])
                return x

        class ParaListGraph(flow.nn.Graph):
            def __init__(self):
                super().__init__()
                self.params = flow.nn.ParameterList(list_of_p)

            def build(self, x):
                for i, _ in enumerate(self.params):
                    x = flow._C.matmul(x, self.params[i])
                return x

        para_list_m = ParaListModule()
        para_list_g = ParaListGraph()
        # print(para_list_g)

        input = flow.tensor(np.random.randn(4, 10), dtype=flow.float32)
        output_m = para_list_m(input)
        # print(output_m)
        output_g = para_list_g(input)

        # print(para_list_g)
        test_case.assertTrue(np.array_equal(output_m.numpy(), output_g.numpy()))

    def test_block_with_para_dict_container(test_case):
        dict_of_p = {
            "0": flow.nn.Parameter(flow.randn(10, 3)),
            "1": flow.nn.Parameter(flow.randn(10, 10)),
        }

        class ParaDictModule(flow.nn.Module):
            def __init__(self):
                super().__init__()
                self.params = flow.nn.ParameterDict(dict_of_p)

            def forward(self, x):
                x = flow._C.matmul(x, self.params["0"])
                return x

        class ParaDictGraph(flow.nn.Graph):
            def __init__(self):
                super().__init__()
                self.params = flow.nn.ParameterDict(dict_of_p)

            def build(self, x):
                x = flow._C.matmul(x, self.params["0"])
                return x

        para_dict_m = ParaDictModule()
        para_dict_g = ParaDictGraph()
        # print(para_dict_g)

        input = flow.tensor(np.random.randn(4, 10), dtype=flow.float32)
        output_m = para_dict_m(input)
        # print(output_m)
        output_g = para_dict_g(input)

        # print(para_dict_g)
        test_case.assertTrue(np.array_equal(output_m.numpy(), output_g.numpy()))

    def test_mixin_module(test_case):
        class ModuleMixin(flow.nn.Module):
            def __init__(self):
                super().__init__()
                self._dtype = flow.float32

            @property
            def dtype(self):
                return self._dtype

        class ConfigMixin:
            def hello_from_cfg(self):
                return "hello_from_cfg"

            @property
            def property_from_cfg(self):
                return 128

        class MixedModule(ModuleMixin, ConfigMixin):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                test_case.assertEqual(self.dtype, flow.float32)
                test_case.assertEqual(self.hello_from_cfg(), "hello_from_cfg")
                test_case.assertEqual(self.property_from_cfg, 128)
                return x

        mixedm = MixedModule()

        class GraphConfigMixin(object):
            @property
            def hello_from_graph(self):
                return "hello_from_gcfg"

            def mixin_get_name(self):
                return self.name

        class MixinGraph(flow.nn.Graph, GraphConfigMixin):
            def __init__(self):
                super().__init__()
                self.m = mixedm

            def build(self, x):
                test_case.assertEqual(self.hello_from_graph, "hello_from_gcfg")
                test_case.assertEqual(self.mixin_get_name(), self.name)
                return self.m(x)

        g = MixinGraph()
        x = np.ones((10, 10))
        x = flow.tensor(x, dtype=flow.float32)
        out = g(x)
        test_case.assertTrue(np.array_equal(x.numpy(), out.numpy()))


if __name__ == "__main__":
    unittest.main()
