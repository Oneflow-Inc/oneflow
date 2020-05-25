# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""
oneflow.python.onnx.rewriter - rewrite tensorflow subgraph to onnx leakyrelu op
"""

from oneflow.python.onnx.graph_matcher import OpTypePattern, GraphMatcher


# pylint: disable=missing-docstring


def rewrite_leakyrelu(g, ops):
    if g.opset < 6:
        return ops

    pattern = \
        OpTypePattern('Maximum', name='max', inputs=[
            OpTypePattern('Mul', name='mul', inputs=[
                OpTypePattern('Const', name='alpha'),
                OpTypePattern('*', name='mul_input'),
            ]),
            OpTypePattern('*', name='max_input'),
        ])

    matcher = GraphMatcher(pattern, allow_reorder=True)
    match_results = list(matcher.match_ops(ops))
    for match in match_results:
        max_node = match.get_op('max')
        max_input_node = match.get_op('max_input')
        mul_node = match.get_op("mul")
        mul_input_node = match.get_op('mul_input')

        max_input_edge_name = _find_edge_name_between_nodes(max_input_node, max_node)
        mul_input_edge_name = _find_edge_name_between_nodes(mul_input_node, mul_node)
        if max_input_edge_name == mul_input_edge_name:
            alpha = match.get_op("alpha").get_tensor_value()
            if alpha >= 1:
                continue
            leakyrelu = g.make_node("LeakyRelu", inputs=[max_input_edge_name], attr={"alpha": alpha},
                                    shapes=[g.get_shape(max_node.output[0])], dtypes=[g.get_dtype(max_node.output[0])])
            ops.append(leakyrelu)
            g.replace_all_inputs(ops, max_node.output[0], leakyrelu.output[0])
            to_delete = [max_node, mul_node]
            g.safe_remove_nodes(to_delete)

    return ops


def _find_edge_name_between_nodes(src_node, consumer_node):
    # find the first edge connection between two nodes.
    for consumer_end in consumer_node.input:
        for src_end in src_node.output:
            if consumer_end == src_end:
                return consumer_end
    return None
