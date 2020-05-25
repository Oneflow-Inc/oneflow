# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""const fold Optimizer.
   if op's inputs are all const then do op computation when building the graph to improve performance
   for example, input of transpose node is const then we can do transpose statically instead of at runtime
"""

from .. import utils
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

    def _optimize(self, graph):
        return self._apply_optimization(graph, self._optimize_at_current_graph_level)

    def _optimize_at_current_graph_level(self, graph):
        graph_changed = True
        while graph_changed:
            graph_changed = False
            ops = graph.get_nodes()
            for op in ops:
                if self._should_skip(op):
                    continue
                if self._fold_node(op, graph):
                    graph_changed = True
                    self.graph_been_opt = True
        return graph

    @staticmethod
    def _should_skip(node):
        # only support onnx official op for now, op in other domain is not supported for now
        if not utils.is_onnx_domain(node.domain):
            return True

        if node.is_const() or node.is_graph_input():
            return True

        skip_type = ["Identity"]
        if node.type in skip_type:
            return True

        return False

    def _fold_node(self, node, graph):
        """ if node's input are all const and it's not graph's output then it can be fold.
            if node can be fold True will be return indicating that graph is changed
        """
        if self._all_inputs_are_const(node.inputs) and not self._is_graph_output(node, graph):
            process_func = _func_map.get(node.type, None)
            if process_func:
                const_outputs = process_func(node, graph)
                self._replace_node_with_const(node, graph, const_outputs)
                return True
            self.logger.debug("need to add function to fold op %s whose op_type is %s", node.name, node.type)
        return False

    @staticmethod
    def _all_inputs_are_const(nodes):
        return all(node.is_const() for node in nodes if node)

    @staticmethod
    def _is_graph_output(node, graph):
        node_out_set = set(node.output)
        graph_out_set = set(graph.outputs)
        return node_out_set.intersection(graph_out_set)

    @staticmethod
    def _replace_node_with_const(node, graph, vals):
        utils.make_sure(len(node.output) == len(vals), "length of node outputs and const vals should be same")
        for old_input, val in zip(node.output, vals):
            const_node = graph.make_const(utils.make_name("const_fold_opt"), val)
            graph.set_dtype(const_node.output[0], utils.map_numpy_to_onnx_dtype(val.dtype))
            graph.set_shape(const_node.output[0], val.shape)
            graph.replace_all_inputs(graph.get_nodes(), old_input, const_node.output[0])
        graph.remove_node(node.name)

    @staticmethod
    @_register_func("Cast")
    def _fold_cast(node, graph):
        const_val = node.inputs[0].get_tensor_value(as_list=False)
        np_dtype = utils.ONNX_TO_NUMPY_DTYPE[node.get_attr("to").i]
        const_val_after_cast = const_val.astype(np_dtype)
        return [const_val_after_cast]

    @staticmethod
    @_register_func("Transpose")
    def _fold_transpose(node, graph) -> list:
        const_val = node.inputs[0].get_tensor_value(as_list=False)
        perm_attr = node.get_attr("perm")
        perm = perm_attr.ints if perm_attr else None
        const_val_after_trans = const_val.transpose(perm)
        return [const_val_after_trans]

    @staticmethod
    @_register_func("Unsqueeze")
    def _fold_unsqueeze(node, graph):
        """
        numpy expand_dims only supports to unsqueeze one dim one time, so reshape is used to simplify the logic
        """
        const_val = node.inputs[0].get_tensor_value(as_list=False)
        axes = list(node.get_attr("axes").ints)
        utils.make_sure(all(axis >= 0 for axis in axes), "onnx spec says it only supports positive axis")
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
