# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""
oneflow.python.onnx.rewriter - rewrite tensorflow transpose op
"""

from oneflow.python.onnx.graph_matcher import OpTypePattern, GraphMatcher


# pylint: disable=missing-docstring


def rewrite_transpose(g, ops):
    pattern = \
        OpTypePattern('Transpose', name='output', inputs=[
            OpTypePattern(None),
            OpTypePattern('Sub', inputs=[
                OpTypePattern('Sub', inputs=["*", "*"]),
                OpTypePattern('Range', inputs=["*", "*", "*"]),
            ]),
        ])

    matcher = GraphMatcher(pattern)
    match_results = list(matcher.match_ops(ops))
    for match in match_results:
        output = match.get_op('output')
        shape = g.get_shape(output.input[0])
        dims = range(len(shape) - 1, -1, -1)
        output.set_attr("perm", dims)
        g.remove_input(output, output.input[1])
        to_delete = [n for n in match.get_nodes() if n != output]
        g.safe_remove_nodes(to_delete)
    return ops
