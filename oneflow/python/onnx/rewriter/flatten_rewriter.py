# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""
oneflow.python.onnx.rewriter - rewrite tensorflow subgraph to onnx flatten op
"""

import numpy as np

from oneflow.python.onnx import utils
from oneflow.python.onnx.graph_matcher import OpTypePattern, GraphMatcher


# pylint: disable=missing-docstring


def rewrite_flatten(g, ops):
    pattern_fixed_shape_input = \
        OpTypePattern('Reshape', name='reshape', inputs=[
            OpTypePattern("*", name="input"),
            OpTypePattern('Pack', name="pack", inputs=[
                OpTypePattern('StridedSlice', name="slice", inputs=[
                    "*", "*", "*", "*",
                ]),
                "*",
            ]),
        ])
    pattern_non_fixed_shape_input = \
        OpTypePattern('Reshape', name='reshape', inputs=[
            OpTypePattern("*", name="input"),
            OpTypePattern('Pack', name="pack", inputs=[
                OpTypePattern('StridedSlice', name="slice", inputs=[
                    OpTypePattern('Shape', inputs=[
                        OpTypePattern("*", name="input2")
                    ]),
                    "*", "*", "*",
                ]),
                "*",
            ]),
        ])
    matcher = GraphMatcher(pattern_fixed_shape_input)
    match_results_1 = list(matcher.match_ops(ops))

    matcher = GraphMatcher(pattern_non_fixed_shape_input)
    match_results_2 = list(matcher.match_ops(ops))

    match_results = [(match_results_1, True), (match_results_2, False)]
    for match_results, check_fixed_input_shape in match_results:
        for match in match_results:
            input_node = match.get_op('input')
            reshape_node = match.get_op('reshape')
            pack_node = match.get_op('pack')
            slice_node = match.get_op('slice')
            need_rewrite = pack_node.inputs[1].is_const() and pack_node.inputs[1].get_tensor_value() == -1
            if not need_rewrite:
                continue

            input_shape = g.get_shape(reshape_node.input[0])
            need_rewrite = input_shape is not None
            if not need_rewrite:
                continue

            if check_fixed_input_shape:
                need_rewrite = slice_node.inputs[0].is_const() and \
                               np.array_equal(list(input_shape), list(slice_node.inputs[0].get_tensor_value()))
                if not need_rewrite:
                    continue

            begin = slice_node.inputs[1].get_tensor_value(as_list=False)
            end = slice_node.inputs[2].get_tensor_value(as_list=False)
            strides = slice_node.inputs[3].get_tensor_value(as_list=False)
            need_rewrite = np.array_equal(begin, [0]) and len(end) == 1 and \
                           np.array_equal(strides, [1]) and end[0] - begin[0] == 1
            if not need_rewrite:
                continue

            to_remove = [n for n in match.get_nodes() if n != input_node]
            safe = g.safe_to_remove_nodes(to_remove)

            # Ok if reshape_node is not safe. Will make it safe later.
            if len(to_remove) - len(safe) > 1:
                continue

            op_name = utils.make_name("Flatten")
            out_name = utils.port_name(op_name)
            g.make_node("Flatten", [reshape_node.input[0]], outputs=[out_name], name=op_name)

            last_dim = input_shape[-1]
            sec_last_dim = input_shape[-2]
            new_dim = None
            if last_dim > 0 and sec_last_dim > 0:
                new_dim = last_dim * sec_last_dim
            else:
                new_dim = -1

            g.set_shape(out_name, input_shape[:-2] + [new_dim])
            g.replace_all_inputs(ops, reshape_node.output[0], out_name)
            for n in to_remove:
                g.remove_node(n.name)

    return ops
