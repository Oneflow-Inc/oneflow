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

# Merge Duplicated Nodes Optimizer.
# duplicate nodes except identity nodes which should be handled by identity optimizer.
# example, node a is input of node b and node c, and computation of node b, c are same such as "abs" op.
# b and c can be merged into one node to avoid duplicated computation

from collections import defaultdict, namedtuple

import numpy as np

from .optimizer_base import GraphOptimizerBase

# pylint: disable=logging-not-lazy,unused-argument,missing-docstring

_KeyToGroupNodes = namedtuple("key", "type input")


class MergeDuplicatedNodesOptimizer(GraphOptimizerBase):
    """Remove duplicate nodes.
    """

    def __init__(self):
        super(MergeDuplicatedNodesOptimizer, self).__init__()
        # used internally
        self._graph_can_be_Optimized = True

    def _Optimize(self, graph):
        return self._ApplyOptimization(graph, self._OptimizeAtCurrentGraphLevel)

    def _OptimizeAtCurrentGraphLevel(self, graph):
        while self._graph_can_be_Optimized:
            self._graph_can_be_Optimized = False
            self._MergeDuplicatedNodes(graph)
            if self._graph_can_be_Optimized:
                self.graph_been_opt = True
        return graph

    def _MergeDuplicatedNodes(self, graph):
        # "duplicated" means: op_type, input and attribute are same
        # while attr is un-hashable so doesn't include it when grouping nodes
        nodes_groups = self._GroupNodesByTypeInputs(graph)
        for _, nodes_group in nodes_groups.items():
            if self._skip_node_type(nodes_group[0]):
                continue
            self._DelNodesIfDuplicated(nodes_group, graph)

    @staticmethod
    def _GroupNodesByTypeInputs(graph):
        res = defaultdict(list)
        for node in graph.get_nodes():
            # default const of graph input cannot be merged
            if node.is_graph_input_default_const():
                continue
            res[_KeyToGroupNodes(node.op_type, tuple(node.input_tensor_names))].append(
                node
            )
        return res

    def _DelNodesIfDuplicated(self, nodes_group, graph):
        # input and op type of nodes in same group are same,
        # and if their attributes are also same then they are duplicated
        while len(nodes_group) > 1:
            unprocessed_node = []
            nodes_to_process = [nodes_group[0]]
            for node in nodes_group[1:]:
                if self._have_equal_attr(node, nodes_to_process[0], graph):
                    nodes_to_process.append(node)
                else:
                    unprocessed_node.append(node)

            self._MergeNodesThatAreDuplicated(nodes_to_process, graph)
            nodes_group = unprocessed_node

    def _have_equal_attr(self, node_1, node_2, graph):
        # above check guarantees consts here are able to be merged
        if node_1.is_const() and node_2.is_const():
            # get_tensor_value is costly so that we check their shape first
            shape_1 = graph.get_shape(node_1.output_tensor_names[0])
            shape_2 = graph.get_shape(node_2.output_tensor_names[0])
            if shape_1 is not None and shape_2 is not None and shape_1 != shape_2:
                return False
            const_1 = node_1.get_tensor_value(as_list=False)
            const_2 = node_2.get_tensor_value(as_list=False)
            if const_1.dtype == const_2.dtype and np.array_equal(const_1, const_2):
                return True
        else:
            if node_1.attrs == node_2.attrs:
                return True
        return False

    def _MergeNodesThatAreDuplicated(self, nodes_to_process, graph):
        # node's output may not all be used, so have to select the one that uses most of node's outputs
        nodes_to_process.sort(key=self._len_of_node_output, reverse=True)
        node_to_retain = nodes_to_process[0]
        for node_to_delete in nodes_to_process[1:]:
            # if one of the output is graph's output then it can't be deleted
            if set(node_to_delete.output_tensor_names).intersection(set(graph.outputs)):
                continue
            for old_input, new_input in zip(
                node_to_delete.output_tensor_names, node_to_retain.output_tensor_names
            ):
                graph.ReplaceAllInputs(graph.get_nodes(), old_input, new_input)
            graph.RemoveNode(node_to_delete.name)
            self._graph_can_be_Optimized = True

    @staticmethod
    def _skip_node_type(node):
        # identity node will be handled by identity optimizer so skip it
        if node.op_type in ["Identity"]:
            return True
        if node.is_graph_input():
            return True
        return False

    @staticmethod
    def _len_of_node_output(node):
        return len(node.output_tensor_names)
