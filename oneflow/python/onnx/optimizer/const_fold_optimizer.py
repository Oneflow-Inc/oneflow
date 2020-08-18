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

# const fold Optimizer.
# if op's inputs are all const then do op computation when building the graph to improve performance
# for example, input of transpose node is const then we can do transpose statically instead of at runtime

from oneflow.python.framework import id_util
from .. import util
from .optimizer_base import GraphOptimizerBase

# pylint: disable=logging-not-lazy,unused-argument,missing-docstring

# key is op_type, value is the function to compute outputs
# the schema of function is: inputs are(node, graph), output is a list of constant values.
_func_map = {}


def _register_func(op_type):
    def _internal_fun(func):
        _func_map[op_type] = func
        return func

    return _internal_fun


class ConstFoldOptimizer(GraphOptimizerBase):
    def __init__(self):  # pylint: disable=useless-super-delegation
        super(ConstFoldOptimizer, self).__init__()

    def _Optimize(self, graph):
        return self._ApplyOptimization(graph, self._OptimizeAtCurrentGraphLevel)

    def _OptimizeAtCurrentGraphLevel(self, graph):
        graph_changed = True
        while graph_changed:
            graph_changed = False
            ops = graph.get_nodes()
            for op in ops:
                if self._ShouldSkip(op):
                    continue
                if self._FoldNode(op, graph):
                    graph_changed = True
                    self.graph_been_opt = True
        return graph

    @staticmethod
    def _ShouldSkip(node):
        # only support onnx official op for now, op in other domain is not supported for now
        if not util.is_onnx_domain(node.domain):
            return True

        if node.is_const() or node.is_graph_input():
            return True

        skip_type = ["Identity"]
        if node.op_type in skip_type:
            return True

        return False

    def _FoldNode(self, node, graph):
        """ if node's input are all const and it's not graph's output then it can be fold.
            if node can be fold True will be return indicating that graph is changed
        """
        if self._AllInputsAreConst(node.input_nodes) and not self._IsGraphOutput(
            node, graph
        ):
            process_func = _func_map.get(node.op_type, None)
            if process_func:
                const_outputs = process_func(node, graph)
                self._ReplaceNodeWithConst(node, graph, const_outputs)
                return True
            self.logger.debug(
                "need to add function to fold op %s whose op_type is %s",
                node.name,
                node.op_type,
            )
        return False

    @staticmethod
    def _AllInputsAreConst(nodes):
        return all(node.is_const() for node in nodes if node)

    @staticmethod
    def _IsGraphOutput(node, graph):
        node_out_set = set(node.output_tensor_names)
        graph_out_set = set(graph.outputs)
        return node_out_set.intersection(graph_out_set)

    @staticmethod
    def _ReplaceNodeWithConst(node, graph, vals):
        util.MakeSure(
            len(node.output_tensor_names) == len(vals),
            "length of node outputs and const vals should be same",
        )
        for old_input, val in zip(node.output_tensor_names, vals):
            const_node = graph.MakeConst(id_util.UniqueStr("const_fold_opt"), val)
            graph.set_dtype(
                const_node.output_tensor_names[0], util.Numpy2OnnxDtype(val.dtype)
            )
            graph.set_shape(const_node.output_tensor_names[0], val.shape)
            graph.ReplaceAllInputs(
                graph.get_nodes(), old_input, const_node.output_tensor_names[0]
            )
        graph.RemoveNode(node.name)

    @staticmethod
    @_register_func("Cast")
    def _FoldCast(node, graph):
        const_val = node.input_nodes[0].get_tensor_value(as_list=False)
        np_dtype = util.ONNX_2_NUMPY_DTYPE[node.attrs["to"]]
        const_val_after_cast = const_val.astype(np_dtype)
        return [const_val_after_cast]

    @staticmethod
    @_register_func("Transpose")
    def _FoldTranspose(node, graph) -> list:
        const_val = node.input_nodes[0].get_tensor_value(as_list=False)
        perm = node.attrs.get("perm", None)
        const_val_after_trans = const_val.transpose(perm)
        return [const_val_after_trans]

    @staticmethod
    @_register_func("Unsqueeze")
    def _FoldUnsqueeze(node, graph):
        """
        numpy expand_dims only supports to unsqueeze one dim one time, so reshape is used to simplify the logic
        """
        const_val = node.input_nodes[0].get_tensor_value(as_list=False)
        axes = node.attrs["axes"]
        util.MakeSure(
            all(axis >= 0 for axis in axes),
            "onnx spec says it only supports positive axis",
        )
        shape_in = const_val.shape
        dims_out = len(shape_in) + len(axes)
        # calculate the shape of output accroding to onnx Unsqueeze's spec
        # https://github.com/onnx/onnx/blob/master/docs/Operators.md#Unsqueeze
        shape_in = iter(shape_in)
        shape_out = [None] * dims_out
        for ind in axes:
            shape_out[ind] = 1
        for ind, val in enumerate(shape_out):
            if val is None:
                shape_out[ind] = next(shape_in)

        const_val_after_unsqueeze = const_val.reshape(shape_out)
        return [const_val_after_unsqueeze]
