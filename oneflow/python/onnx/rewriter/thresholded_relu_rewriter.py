# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""
oneflow.python.onnx.rewriter - rewrite tensorflow subgraph to onnx ThresholdedRelu op
"""

from oneflow.python.onnx.graph_matcher import OpTypePattern, GraphMatcher
from oneflow.python.onnx.rewriter.leakyrelu_rewriter import _find_edge_name_between_nodes


# pylint: disable=missing-docstring


def rewrite_thresholded_relu(g, ops):
    if g.opset < 10:
        return ops

    pattern = \
        OpTypePattern('Mul', name='mul', inputs=[
            OpTypePattern('Cast', name='cast', inputs=[
                OpTypePattern('Greater', name='greater', inputs=[
                    OpTypePattern('*', name='greater_input'),
                    OpTypePattern('Const', name='theta')
                ])
            ]),
            OpTypePattern('*', name='mul_input')
        ])
    matcher = GraphMatcher(pattern, allow_reorder=True)
    match_results = list(matcher.match_ops(ops))

    for match in match_results:
        greater_node = match.get_op('greater')
        greater_input_node = match.get_op('greater_input')
        mul_node = match.get_op("mul")
        mul_input_node = match.get_op('mul_input')
        cast_node = match.get_op('cast')

        greater_input_edge_name = _find_edge_name_between_nodes(greater_input_node, greater_node)
        mul_input_edge_name = _find_edge_name_between_nodes(mul_input_node, mul_node)
        if greater_input_edge_name == mul_input_edge_name:
            theta = match.get_op('theta').get_tensor_value()
            thresholded_relu = g.make_node("ThresholdedRelu", inputs=[mul_input_edge_name], attr={"alpha": theta},
                                           shapes=[g.get_shape(mul_node.output[0])],
                                           dtypes=[g.get_dtype(mul_node.output[0])])
            g.replace_all_inputs(ops, mul_node.output[0], thresholded_relu.output[0])
            to_delete = [cast_node, mul_node]
            g.safe_remove_nodes(to_delete)
    return ops
