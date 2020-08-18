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

# Back_To_Back Optimizer.
# Collapse consecutive nodes into 1 node if possible.

from __future__ import unicode_literals

from oneflow.python.onnx.util import ONNX_DTYPE_NAMES  # lgtm[py/unsafe-cyclic-import]
from .optimizer_base import GraphOptimizerBase  # lgtm[py/unsafe-cyclic-import]

# pylint: disable=logging-not-lazy,unused-argument,missing-docstring,unused-variable,arguments-differ

_func_map = {}


def _register_func(op_type):
    def _internal_fun(func):
        _func_map[op_type] = func
        return func

    return _internal_fun


class BackToBackOptimizer(GraphOptimizerBase):
    """Remove back-to-back nodes e.g. 'Cast'
    """

    def __init__(self):  # pylint: disable=useless-super-delegation
        super(BackToBackOptimizer, self).__init__()

    def _Optimize(self, graph):
        return self._ApplyOptimization(graph, self._OptimizeAtCurrentGraphLevel)

    def _OptimizeAtCurrentGraphLevel(self, g):
        for optype, handler in _func_map.items():
            # candidate nodes for removal/optimization
            nodes = [n for n in g.get_nodes() if n.op_type in optype]

            # topological sort of candidates
            # simplifying assumption for back-to-back-optimizer is
            # the op_types have 1 input, 1 output, but multiple consumers
            has_dependencies = set()
            consumer_node_ids = {n.output_tensor_names[0]: [] for n in nodes}
            for n in nodes:
                if n.input_tensor_names[0] in consumer_node_ids:
                    consumer_node_ids[n.input_tensor_names[0]].extend([n])
                    has_dependencies.add(n.output_tensor_names[0])

            # q = starting nodes with no dependencies
            q = list(set(consumer_node_ids.keys()) - has_dependencies)
            while q:
                nodeid = q.pop(0)
                node = g.get_node_by_output(nodeid, False)
                consumer_nodes = consumer_node_ids[nodeid]

                if len(consumer_nodes) > 0:
                    all_consumers = g.FindOutputConsumers(node.output_tensor_names[0])
                    if len(all_consumers) != len(consumer_nodes):
                        # if first node is used elsewhere, skip
                        continue
                    if set(node.output_tensor_names) & set(g.outputs):
                        # if this node is part of graph outputs, skip
                        continue
                    q2 = handler(g, node, consumer_nodes)
                    # add more nodes which can now be processed
                    q.extend(q2)
        return g

    @staticmethod
    @_register_func("Cast")
    def _OptimizeCast(g, node, consumer_nodes):
        """remove long chains of cast ops"""
        q2 = []
        type1 = node.attrs["to"]
        type1_name = ONNX_DTYPE_NAMES[type1] if type1 in ONNX_DTYPE_NAMES else ""

        # if parent node is cast node, and same type, delete this one
        pnode = node.input_nodes[0]
        if pnode.op_type == "Cast":
            type2 = pnode.attrs["to"]
            if type1 == type2:
                for node2 in consumer_nodes:
                    node2.input_tensor_names[0] = node.input_tensor_names[0]
                    q2.append(node2.output_tensor_names[0])
                g.RemoveNode(node.name)
                return q2

        # otherwise, check consumer cast nodes for a target type
        # that contains more information than current type
        can_reduce = True
        for node2 in consumer_nodes:
            type2 = node2.attrs["to"]
            type2_name = ONNX_DTYPE_NAMES[type2] if type2 in ONNX_DTYPE_NAMES else ""

            if "float" in type1_name or type1_name == "double":
                # high information type. ok to eliminate
                pass
            elif "int" in type1_name:
                # int* and uint* are mix of high and low information.
                # for safety, keep the current node, unless type2 is bool,
                # in which case it's ok to remove node
                if type1 != type2 and type2_name != "bool":
                    can_reduce = False
            elif type1_name == "bool":
                # bool is low information, so don't eliminate
                if type1 != type2:
                    can_reduce = False
            elif type1_name == "string":
                # can always remove string
                pass
            else:
                # some odd type, keep node
                can_reduce = False
            q2.append(node2.output_tensor_names[0])

        if can_reduce:
            for node2 in consumer_nodes:
                node2.input_tensor_names[0] = node.input_tensor_names[0]
            g.RemoveNode(node.name)
        return q2

    @staticmethod
    @_register_func("Transpose")
    def _OptimizeTranspose(g, node, consumer_nodes):
        """remove long chains of transpose ops"""
        t1 = node.attrs["perm"]
        q2 = []
        for node2 in consumer_nodes:
            node2.input_tensor_names[0] = node.input_tensor_names[0]
            t2 = node.attrs["perm"]
            new_perm = [t1[i] for i in t2]
            # check if node2 can be removed. otherwise only update
            if new_perm == list(range(len(t2))):
                # both nodes can be deleted
                shape = g.get_shape(node2.output_tensor_names[0])
                dtype = g.get_dtype(node2.output_tensor_names[0])
                node2_consumers = g.FindOutputConsumers(node2.output_tensor_names[0])
                g.ReplaceAllInputs(
                    node2_consumers,
                    node2.output_tensor_names[0],
                    node.input_tensor_names[0],
                )
                g.RemoveNode(node2.name)
                if set(node2.output_tensor_names) & set(g.outputs):
                    g.MakeNode(
                        "Identity",
                        [node.input_tensor_names[0]],
                        outputs=node2.output_tensor_names,
                        shapes=[shape],
                        dtypes=[dtype],
                    )
            else:
                node2.attrs["perm"] = [t1[i] for i in t2]
                q2.append(node2.output_tensor_names[0])
        g.RemoveNode(node.name)
        return q2

    @staticmethod
    @_register_func(("Squeeze", "Unsqueeze"))
    def _OptimizeSqueezeUnsqueeze(g, node, consumer_nodes):
        """remove pairs of squeeze-unsqueeze nodes"""

        if node.op_type != "Squeeze" or len(consumer_nodes) != 1:
            # no need to return any value, since not removing long chain of nodes
            return []

        node2 = consumer_nodes[0]
        if node2.op_type != "Unsqueeze":
            return []

        axis1 = node.attrs["axes"]
        axis2 = node2.attrs["axes"]

        # if squeeze followed by unsqueeze is on diff axes, skip
        if axis1 != axis2:
            return []

        # if unsqueeze output is graph output, skip
        if set(node2.output_tensor_names) & set(g.outputs):
            return []

        node2_consumers = g.FindOutputConsumers(node2.output_tensor_names[0])
        g.ReplaceAllInputs(
            node2_consumers, node2.output_tensor_names[0], node.input_tensor_names[0]
        )
        g.RemoveNode(node.name)
        g.RemoveNode(node2.name)
        return []
