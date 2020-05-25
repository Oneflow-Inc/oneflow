# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""
oneflow.python.onnx.rewrite - rewrite tensorflow subgraph to onnx gemm op
"""
import logging
from onnx import onnx_pb
from oneflow.python.onnx.graph_matcher import OpTypePattern, GraphMatcher

# pylint: disable=missing-docstring

def rewrite_gemm(g, ops):
    if g.opset <= 6:
        return ops

    # pattern0: alpha*A*B + beta*C
    pattern0 = \
        OpTypePattern('Add|AddV2', name='add', inputs=[
            OpTypePattern('Mul', name='mul1', inputs=[
                OpTypePattern('Const', name='alpha'),
                OpTypePattern('MatMul', name='matmul')
            ]),
            OpTypePattern('Mul', name='mul2', inputs=[
                OpTypePattern('Const', name='beta'),
                OpTypePattern('*', name='C')
            ])
        ])

    # pattern1: alpha*A*B + C
    pattern1 = \
        OpTypePattern('Add|AddV2', name='add', inputs=[
            OpTypePattern('Mul', name='mul1', inputs=[
                OpTypePattern('MatMul', name='matmul'),
                OpTypePattern('Const', name='alpha')
            ]),
            OpTypePattern('*', name='C'),
        ])

    # pattern2: A*B + beta*C
    pattern2 = \
        OpTypePattern('Add|AddV2', name='add', inputs=[
            OpTypePattern('MatMul', name='matmul'),
            OpTypePattern('Mul', name='mul2', inputs=[
                OpTypePattern('Const', name='beta'),
                OpTypePattern('*', name='C')
            ])
        ])

    # pattern3: A*B + C
    pattern3 = \
        OpTypePattern('Add|AddV2', name='add', inputs=[
            OpTypePattern('MatMul', name='matmul'),
            OpTypePattern('*', name='C'),
        ])

    pattern_list = [pattern0, pattern1, pattern2, pattern3]

    for pattern in pattern_list:
        matcher = GraphMatcher(pattern, allow_reorder=True)
        match_results = list(matcher.match_ops(ops))
        if match_results:
            for match in match_results:
                matmul_node = match.get_op("matmul")

                if g.get_dtype(matmul_node.input[0]) != onnx_pb.TensorProto.FLOAT:
                    logging.warning(u"For now, onnxruntime only support float32 type for Gemm rewriter")
                    continue

                attr, is_valid = get_gemm_attr(match)
                if not is_valid:
                    continue

                add_node = match.get_op('add')
                input_c_node = match.get_op("C")
                a_edge_name = matmul_node.input[0]
                b_edge_name = matmul_node.input[1]
                c_edge_name = input_c_node.output[0]

                gemm = g.make_node("Gemm", inputs=[a_edge_name, b_edge_name, c_edge_name],
                                   attr=attr,
                                   shapes=[g.get_shape(add_node.output[0])],
                                   dtypes=[g.get_dtype(add_node.output[0])])

                ops.append(gemm)
                g.replace_all_inputs(ops, add_node.output[0], gemm.output[0])
                to_delete = [add_node, matmul_node]
                g.safe_remove_nodes(to_delete)
    return ops

def get_gemm_attr(match):
    attr = {}
    for arg in ["alpha", "beta"]:
        arg_op = match.get_op(arg)
        if arg_op is not None:
            match_args = arg_op.get_tensor_value()
            if isinstance(match_args, list):
                if len(match_args) != 1:
                    return attr, False
                match_args = match_args[0]
            attr[arg] = match_args
    return attr, True
