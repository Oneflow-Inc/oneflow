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

# Identity Optimizer.
# useless Identity node in graphs including subgraphs, but does not hurt model output names.

from __future__ import unicode_literals

from .optimizer_base import GraphOptimizerBase


# pylint: disable=logging-not-lazy,unused-argument,missing-docstring,unused-variable,arguments-differ


class IdentityOptimizer(GraphOptimizerBase):
    """Identity Optimizer."""

    def __init__(self):  # pylint: disable=useless-super-delegation
        super(IdentityOptimizer, self).__init__()

    def _Optimize(self, graph):
        return self._ApplyOptimization(graph, self._OptimizeAtCurrentGraphLevel)

    def _OptimizeAtCurrentGraphLevel(self, g):
        has_update = True
        while has_update:
            has_update = False
            nodes = [n for n in g.get_nodes() if n.op_type == "Identity"]
            for n in nodes:
                if n.graph is None:
                    self.logger.debug("node has been removed from this graph, skip")
                    continue

                graph_outputs = set(n.output_tensor_names).intersection(g.outputs)
                ret = False
                if graph_outputs:
                    ret = self._HandleGraphOutputIdentity(g, n, graph_outputs)
                else:
                    ret = self._HandleNonGraphOutputIdentity(g, n)
                has_update = ret
                if ret:
                    self.graph_been_opt = True
        return g

    @staticmethod
    def _HandleNonGraphOutputIdentity(graph, identity):
        graph.ReplaceAllInputs(
            graph.get_nodes(),
            identity.output_tensor_names[0],
            identity.input_tensor_names[0],
        )
        graph.RemoveNode(identity.name)
        return True

    def _HandleGraphOutputIdentity(self, graph, identity, graph_outputs):
        input_id = identity.input_tensor_names[0]
        input_node = identity.input_nodes[0]

        if input_node.graph != graph:
            # If input node is in parent graph, we don't handle it now
            self.logger.debug("input node in parent graph, skip")
            return False

        if input_node.is_graph_input():
            # Identity between input and output should not be removed.
            self.logger.debug("skip identity between input and output")
            return False

        output_id = identity.output_tensor_names[0]
        output_shape = graph.get_shape(output_id)
        output_dtype = graph.get_dtype(output_id)
        if input_id in graph.outputs:
            # input id already be graph output, so we cannot make that be another graph output.
            # this Identity must be kept.
            self.logger.debug("identity input already be graph output")
            return False

        graph.RemoveNode(identity.name)
        new_output = [
            output_id if o == input_id else o for o in input_node.output_tensor_names
        ]
        input_node.output_tensor_names = new_output

        graph.set_shape(output_id, output_shape)
        graph.set_dtype(output_id, output_dtype)

        graph.ReplaceAllInputs(graph.get_nodes(), input_id, output_id)
        return True
