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
import re
import unittest

import numpy as np
import oneflow as flow
import oneflow.unittest
from alexnet_model import alexnet


class TvmFrontedGraph(flow.nn.Graph):
    def __init__(self, module):
        super().__init__()
        self.m = module

    def build(self, x):
        out = self.m(x)
        return out


def parse_attr(attr):
    # Parse node_attr
    attrs = {}
    for a in attr:
        attr_str = str(attr[a])

        if attr_str[0:7] == "at_list":
            attr_str_ = attr_str.split(" ")[0]

            if attr_str_ == "at_list_float":
                attrs[a] = tuple(attr[a].at_list_float.val)
            elif attr_str_ == "at_list_int32":
                attrs[a] = tuple(attr[a].at_list_int32.val)
            elif attr_str_ == "at_list_int64":
                attrs[a] = tuple(attr[a].at_list_int64.val)

        elif attr_str.split(":")[0] == "at_string":
            attrs[a] = attr[a].at_string

        elif attr_str.split(" ")[0] == "at_shape":
            attrs[a] = tuple(list(attr[a].at_shape.dim))

        else:
            attr_str_ = attr_str.split(":")[0]
            if attr_str_ == "at_bool":
                attrs[a] = attr[a].at_bool
            elif attr_str_ == "at_double":
                attrs[a] = attr[a].at_double
            elif attr_str_ == "at_float":
                attrs[a] = attr[a].at_float
            elif attr_str_ == "at_int32":
                attrs[a] = attr[a].at_int32
            elif attr_str_ == "at_int64":
                attrs[a] = attr[a].at_int64

    return attrs


def is_user_op(node):
    # Determine if the the node is the intermediate variables of graph
    return node.WhichOneof("op_type") == "user_conf"


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
@flow.unittest.skip_unless_1n1d()
class TestConvertDependency(flow.unittest.TestCase):
    def test_get_params(test_case):
        class ConvModel(flow.nn.Module):
            def __init__(self):
                super(ConvModel, self).__init__()
                self.conv = flow.nn.Conv2d(3, 64, kernel_size=11, bias=False)

            def forward(self, x):
                x = self.conv(x)
                return x

        model = ConvModel().state_dict()
        for layer_name in model:
            layer_path = os.path.join(layer_name, "out")
            test_case.assertEqual(layer_path, "conv.weight/out")

    def test_infos_of_nodes(test_case):
        alexnet_module = alexnet()
        alexnet_graph = TvmFrontedGraph(alexnet_module)
        if not alexnet_graph._is_compiled:
            alexnet_graph._compile(flow.rand(1, 3, 224, 224))
        graph_str = repr(alexnet_graph)
        if not alexnet_graph._is_compiled:
            alexnet_graph._compile(flow.rand(shape_input))

        size_where = 2
        if "cuda" in graph_str:
            size_where = 3

        p_size = re.compile(r"size=\(.*?\)", re.S)
        p_type = re.compile(r"(dtype=.*?)[,|\)]", re.S)
        types = ["INPUT", "PARAMETER", "BUFFER", "OUTPUT"]
        num_nodes = {}

        for t in types:
            data = re.finditer(t + ":.*", graph_str)
            cnt = 0
            for i in data:
                cnt += 1
                attrs = i.group().split(":")
                size_strs = re.findall(p_size, attrs[size_where])
                type_strs = re.findall(p_type, attrs[size_where])
                test_case.assertEqual(size_strs != [], True)
                test_case.assertEqual(type_strs != [], True)

                size_attr = size_strs[0].replace("size=", "")
                type_attr = type_strs[0].replace("dtype=", "").replace(")", "")
                if size_attr[-2] == ",":
                    size_attr = size_attr.replace(",", "")
                if type_attr[-1] == ",":
                    type_attr = type_attr.replace(",", "")
                    test_case.assertEqual(type_attr, "oneflow.float32")

                data_size = tuple(map(int, size_attr[1:-1].split(", ")))
                if cnt == 1 and t == "PARAMETER":
                    test_case.assertEqual(data_size, (64, 3, 11, 11))
                elif cnt == 15 and t == "PARAMETER":
                    test_case.assertEqual(data_size, (1000, 4096))
            num_nodes[t] = cnt

        test_case.assertEqual(num_nodes["INPUT"] != 0, True)
        test_case.assertEqual(num_nodes["BUFFER"], 0)
        test_case.assertEqual(num_nodes["PARAMETER"], 16)
        test_case.assertEqual(num_nodes["OUTPUT"] != 0, True)

        # get graph proto, if you don't _compile the graph, the _graph_proto will be None
        graph_input = re.search(r"INPUT:.*", graph_str).group().split(":")
        shape_input = tuple(
            map(
                int,
                re.findall(p_size, graph_input[size_where])[0]
                .replace("size=", "")[1:-1]
                .split(", "),
            )
        )
        graph_proto = alexnet_graph._graph_proto

        nodes = {}
        for op in graph_proto.net.op:
            nodes[op.name] = op

        op_names = []
        op_attrs = []
        for node_name in nodes:
            node = nodes[node_name]
            if is_user_op(node):
                op_name = node.user_conf.op_type_name
                op_attr = parse_attr(node.user_conf.attr)
                op_names.append(op_name)
                op_attrs.append(op_attr)

        test_case.assertEqual(op_names[0], "conv2d")
        test_case.assertEqual(op_names[1], "bias_add")
        test_case.assertEqual(op_names[2], "relu")

        kernel_size = op_attrs[0].get("kernel_size", None)
        strides = op_attrs[0].get("strides", None)
        padding_before = op_attrs[0].get("padding_before", None)
        test_case.assertEqual(kernel_size, (11, 11))
        test_case.assertEqual(strides, (4, 4))
        test_case.assertEqual(padding_before, (2, 2))

        node_input_list = []
        node_output_list = []
        for node_name in nodes:
            node = nodes[node_name]
            if is_user_op(node) and node.user_conf.op_type_name == "conv2d":
                for input_name in node.user_conf.input:
                    node_input_paths = getattr(node.user_conf.input[input_name], "s")
                    for i in node_input_paths:
                        node_input = i.split("/")[0]
                        print(node_input)
                        node_input_list.append(node_input)
                for output_name in node.user_conf.output:
                    node_output_paths = getattr(node.user_conf.output[output_name], "s")
                    for node_output_path in node_output_paths:
                        node_output_name = node_output_path.split("/")[0]
                        print(node_output_name)
                        node_output_list.append(node_output_name)

        test_case.assertEqual("_TvmFrontedGraph_1_input.0.0_2" in node_input_list, True)
        test_case.assertEqual("m.features.0.weight" in node_input_list, True)
        test_case.assertEqual("m.features.5-max_pool_2d-7" in node_input_list, True)
        test_case.assertEqual("m.features.0-conv2d-0" in node_output_list, True)
        test_case.assertEqual("m.features.6-conv2d-8" in node_output_list, True)

    def test_buffer_convert_dependence(test_case):
        class SubModule(flow.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = flow.nn.Linear(36, 4, False)
                self.register_buffer("dummy_buff", flow.Tensor(1, 4))

            def forward(self, x):
                x = self.fc1(x)
                x += self.dummy_buff
                return x

        sub_module = SubModule()
        sub_graph = TvmFrontedGraph(sub_module)
        graph_str = repr(sub_graph)

        size_where = 2
        if "cuda" in graph_str:
            size_where = 3

        p_size = re.compile(r"size=\(.*?\)", re.S)
        p_type = re.compile(r"dtype=.*?,", re.S)
        num_nodes = {}

        data = re.finditer("BUFFER:.*", graph_str)
        for i in data:
            attrs = i.group().split(":")
            size_strs = re.findall(p_size, attrs[size_where])
            size_attr = size_strs[0].replace("size=", "")
            data_size = tuple(map(int, size_attr[1:-1].split(", ")))
            test_case.assertEqual(data_size, (1, 4))


if __name__ == "__main__":
    unittest.main()
