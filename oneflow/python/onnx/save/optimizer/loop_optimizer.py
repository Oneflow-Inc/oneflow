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
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

# Loop Optimizer.
# op in loop's body graph can be moved out to the loop

from oneflow.python.framework import id_util
from oneflow.python.onnx.util import MakeSure
from .optimizer_base import GraphOptimizerBase


# pylint: disable=logging-not-lazy,unused-argument,missing-docstring,unused-variable,arguments-differ


class LoopOptimizer(GraphOptimizerBase):
    """Loop Optimizer."""

    # a lot of terms used here come from loop's onnx spec
    # https://github.com/onnx/onnx/blob/master/docs/Operators.md#Loop
    def __init__(self):  # pylint: disable=useless-super-delegation
        super(LoopOptimizer, self).__init__()

    def _Optimize(self, graph):
        return self._ApplyOptimization(graph, self._OptimizeAtCurrentGraphLevel)

    def _OptimizeAtCurrentGraphLevel(self, g):
        has_update = True
        while has_update:
            has_update = False
            nodes = [n for n in g.get_nodes() if n.op_type == "Loop"]
            for n in nodes:
                has_update_tmp = self._TryMoveTransposeOutOfBodyGraph(n)
                if has_update_tmp:
                    has_update = True
                    self.graph_been_opt = True
        return g

    @staticmethod
    def ConsumerNodesNum(graph, node):
        MakeSure(
            len(node.output_tensor_names) == 1,
            "only consider node with only one output",
        )
        res = len(graph.FindOutputConsumers(node.output_tensor_names[0]))
        return res

    def _TryMoveTransposeOutOfBodyGraph(self, loop_node):
        # output node of body graph can be loop-carried-dependent, if so it can't be move out of the body graph
        # return True if moving some nodes successfully
        # for now, we only consider moving transpose
        body_graph = loop_node.get_body_graphs()["body"]
        parent_graph = loop_node.graph
        scan_nodes_name_in_body, scan_node_in_parent = self._ScanOutputs(loop_node)
        scan_nodes = [
            body_graph.get_node_by_output(name) for name in scan_nodes_name_in_body
        ]
        graph_is_changed = False
        for node, name_in_parent in zip(scan_nodes, scan_node_in_parent):
            # 1 delete node in body graph if possible
            # only consider two case: trans is output, or transpose > identity > output
            need_process = False
            if (
                node.op_type == "Transpose"
                and self.ConsumerNodesNum(body_graph, node) <= 1
            ):
                trans = node
                new_output = node.input_tensor_names[0]
                body_graph.RemoveNode(node.name)
                need_process = True
            elif (
                node.op_type == "Identity"
                and node.input_nodes[0].op_type == "Transpose"
                and self.ConsumerNodesNum(body_graph, node) <= 1
                and self.ConsumerNodesNum(body_graph, node.input_nodes[0]) <= 1
            ):
                trans = node.input_nodes[0]
                new_output = node.input_nodes[0].input_tensor_names[0]
                body_graph.RemoveNode(node.input_nodes[0].name)
                body_graph.RemoveNode(node.name)
                need_process = True

            if need_process:
                # 2 correct body graph's output
                body_outputs = body_graph.outputs
                body_outputs[
                    body_outputs.index(node.output_tensor_names[0])
                ] = new_output
                # 3 insert new node in parent graph
                ori_perm = trans.attrs["perm"]
                new_perm = [0] + [
                    i + 1 for i in ori_perm
                ]  # body output's rank is m > rank of loop's output is m+1
                name = id_util.UniqueStr("trans_moved_from_loop_body")
                _ = parent_graph.InsertNewNodeOnOutput(
                    "Transpose", name_in_parent, name, perm=new_perm
                )
                graph_is_changed = True

        return graph_is_changed

    @classmethod
    def _ScanOutputs(cls, loop):
        # loop has 2+N inputs; loop has N+K outputs;
        # loop's body graph has 1+N+K outputs
        loop_carried = len(loop.input_tensor_names) - 2
        body_graph = loop.get_body_graphs()["body"]
        return (
            body_graph.outputs[loop_carried + 1 :],
            loop.output_tensor_names[loop_carried:],
        )
