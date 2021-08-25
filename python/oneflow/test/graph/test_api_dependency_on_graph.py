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

import oneflow
import oneflow as flow
import oneflow.unittest
from alexnet_model import alexnet


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
@flow.unittest.skip_unless_1n1d()
class TestConvertDependency(flow.unittest.TestCase):
    def test_get_params(test_case):
        model_dir_path = "alexnet_oneflow_model"
        model = flow.load(model_dir_path)
        for layer_name in model:
            layer = model[layer_name]
            layer_path = layer.file_path     # get path
            test_case.assertEqual(layer_path!=None, True)


    def test_infos_of_nodes(test_case):
        class Graph(flow.nn.Graph):
            def __init__(self, module):
                self.m = module
            
            def build(self, x):
                out = self.m(x)
                return out

        alexnet_module = alexnet()
        alexnet_graph = Graph(alexnet_module)

        graph_str = repr(alexnet_graph)
        size_where = 2
        if "cuda" in graph_str:
            size_where = 3

        p_size = re.compile(r"size=\(.*?\)", re.S)
        p_type = re.compile(r"dtype=.*?,", re.S)
        types = ["INPUT", "PARAMETER", "BUFFER", "OUTPUT"]
        num_nodes = {}

        for t in types:
            data = re.finditer(t+":.*", graph_str)
            cnt = 0
            for i in data:
                cnt += 1
                attrs = i.group().split(":")
                size_strs = re.findall(p_size, attrs[size_where])
                type_strs = re.findall(p_type, attrs[size_where])
                test_case.assertEqual(size_strs!=[], True)
                test_case.assertEqual(type_strs!=[], True)

                size_attr = size_strs[0].replace("size=", "")
                type_attr = type_strs[0].replace("dtype=", "")
                if size_attr[-2] == ",":
                    size_attr = size_attr.replace(",", "")
                if type_attr[-1] == ",":
                    type_attr = type_attr.replace(",", "")
                    test_case.assertEqual(type_attr, "oneflow.float32")

                data_size = tuple(map(int, size_attr[1:-1].split(", ")))
                node_name = attrs[1]
            num_nodes[t] = cnt

        test_case.assertEqual(num_nodes["INPUT"]!=0, True)
        test_case.assertEqual(num_nodes["PARAMETER"]==16, True)
        test_case.assertEqual(num_nodes["OUTPUT"]!=0, True)

        # get graph proto, if you don't _compile the graph, the _graph_proto will be None
        graph_input = re.search(r"INPUT:.*", graph_str).group().split(":")
        shape_input = tuple(
            map(
                int, re.findall(
                    p_size, graph_input[size_where]
                )[0].replace("size=", "")[1:-1].split(", ")
            )
        )
        if not graph._is_compiled:
            _ = graph._compile(flow.rand(shape_input))
        graph_proto = graph._graph_proto

        nodes = {}
        for op in graph_proto.net.op:
            nodes[op.name] = op
        print(nodes)


if __name__ == "__main__":
    unittest.main()
