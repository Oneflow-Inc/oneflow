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

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import logging

import numpy as np
from onnx import onnx_pb
from onnx import TensorProto
from oneflow.python.framework import id_util
from oneflow.python.onnx import constants, util
from oneflow.python.onnx.handler import flow_op
from oneflow.python.onnx.onnx_opset import common

logger = logging.getLogger(__name__)


# pylint: disable=unused-argument,missing-docstring


@flow_op(["broadcast_add", "scalar_add_by_tensor"], onnx_op="Add")
@flow_op(["broadcast_sub", "scalar_sub_by_tensor"], onnx_op="Sub", flow_ibns=["x", "y"])
@flow_op(["multiply", "broadcast_mul", "scalar_mul_by_tensor"], onnx_op="Mul")
@flow_op(["broadcast_div", "scalar_div_by_tensor"], onnx_op="Div", flow_ibns=["x", "y"])
class BroadcastOp(common.BroadcastOp):
    pass


@flow_op("scalar_mul", "Mul")
@flow_op("scalar_add", "Add")
class ScalarBinaryOp:
    @classmethod
    def Version_6(cls, ctx, node, **kwargs):
        scalar_val = (
            node.attrs["int_operand"]
            if node.attrs["has_int_operand"]
            else node.attrs["float_operand"]
        )
        np_dtype = util.Onnx2NumpyDtype(ctx.get_dtype(node.input_tensor_names[0]))
        scalar_node = ctx.MakeConst(
            id_util.UniqueStr("scalar"), np.array([scalar_val]).astype(np_dtype)
        )
        node.input_tensor_names.append(scalar_node.output_tensor_names[0])


@flow_op("add_n", onnx_op="Sum")
class AddN:
    @classmethod
    def Version_1(cls, ctx, node, **kwargs):
        pass


@flow_op("bias_add", onnx_op="Add", flow_ibns=["a", "b"])
class BiasAdd(common.BroadcastOp):
    @classmethod
    def Version_6(cls, ctx, node, **kwargs):
        axis = node.attrs["axis"]
        unsqueeze_axes = []
        x_rank = len(ctx.get_shape(node.input_tensor_names[0]))
        for i in range(x_rank):
            if axis != i:
                unsqueeze_axes.append(i)
        unsqueeze_shape = [1] * x_rank
        assert len(ctx.get_shape(node.input_tensor_names[1])) == 1
        unsqueeze_shape[axis] = ctx.get_shape(node.input_tensor_names[1])[0]
        unsqueeze_dtype = ctx.get_dtype(node.input_tensor_names[1])
        ctx.InsertNewNodeOnInput(
            node, "Unsqueeze", node.input_tensor_names[1], axes=unsqueeze_axes
        )
        ctx.set_shape(node.input_tensor_names[1], unsqueeze_shape)
        ctx.set_dtype(node.input_tensor_names[1], unsqueeze_dtype)
        super().Version_6(ctx, node, **kwargs)


@flow_op(["leaky_relu", "softplus"], onnx_op=["LeakyRelu", "Softplus"])
class DirectOpSinceOpset1:
    @classmethod
    def Version_1(cls, ctx, node, **kwargs):
        pass


@flow_op(
    [
        "abs",
        "ceil",
        "elu",
        "exp",
        "floor",
        "log",
        "neg",
        "sigmoid",
        "sigmoid_v2",
        "sqrt",
        "tanh",
        "reciprocal",
        "relu",
    ],
    [
        "Abs",
        "Ceil",
        "Elu",
        "Exp",
        "Floor",
        "Log",
        "Neg",
        "Sigmoid",
        "Sigmoid",
        "Sqrt",
        "Tanh",
        "Reciprocal",
        "Relu",
    ],
)
class DirectOp:
    @classmethod
    def Version_1(cls, ctx, node, **kwargs):
        pass

    @classmethod
    def Version_6(cls, ctx, node, **kwargs):
        pass


@flow_op(
    ["acos", "asin", "atan", "cos", "sin", "tan"],
    ["Acos", "Asin", "Atan", "Cos", "Sin", "Tan"],
)
class TrigOpSinceOpset7:
    @classmethod
    def Version_7(cls, ctx, node, **kwargs):
        pass


@flow_op(
    ["acosh", "asinh", "atanh", "cosh", "sinh"],
    ["Acosh", "Asinh", "Atanh", "Cosh", "Sinh"],
)
class TrigOpSinceOpset9:
    @classmethod
    def Version_9(cls, ctx, node, **kwargs):
        pass


def _MakeMinOrMaxOp(
    ctx, op_type, inputs, outputs, output_shapes=None, output_dtypes=None
):
    # support more dtype
    supported_dtypes = [
        onnx_pb.TensorProto.FLOAT,
        onnx_pb.TensorProto.FLOAT16,
        onnx_pb.TensorProto.DOUBLE,
    ]
    target_dtype = onnx_pb.TensorProto.FLOAT
    need_cast = False
    cast_inputs = []
    for inp in inputs:
        dtype = ctx.get_dtype(inp)
        util.MakeSure(dtype is not None, "dtype of {} is None".format(inp))
        if dtype not in supported_dtypes:
            cast_inp = ctx.MakeNode("Cast", [inp], attr={"to": target_dtype})
            cast_inputs.append(cast_inp.output_tensor_names[0])
            need_cast = True
        else:
            cast_inputs.append(inp)
    node = ctx.MakeNode(op_type, cast_inputs, shapes=output_shapes)
    actual_outputs = node.output_tensor_names
    if need_cast:
        origin_dtype = ctx.get_dtype(inputs[0])
        if output_dtypes is not None:
            origin_dtype = output_dtypes[0]
        ctx.set_dtype(node.output_tensor_names[0], target_dtype)
        cast_name = id_util.UniqueStr(node.name)
        cast_node = ctx.InsertNewNodeOnOutput(
            "Cast", node.output_tensor_names[0], name=cast_name, to=origin_dtype
        )
        ctx.set_dtype(cast_node.output_tensor_names[0], origin_dtype)
        ctx.CopyShape(node.output_tensor_names[0], cast_node.output_tensor_names[0])
        actual_outputs = cast_node.output_tensor_names
    ctx.MakeNode(
        "Identity",
        actual_outputs,
        outputs=outputs,
        shapes=output_shapes,
        dtypes=output_dtypes,
    )

    # onnx < opset 8 does not support broadcasting
    # handle this by doing something like:
    # y = min(x1, add(x2, sub(x1, x1))), where x1, x2 are the inputs and x2 is a scalar
    # this will create a tensor of zeros of the shape of x1, adds x2 to it (which broadcasts) and use that for min.
    shapeo = ctx.get_shape(node.output_tensor_names[0])
    needs_broadcast_op = []
    has_correct_shape = []
    if ctx.opset < 8:
        for i, input_name in enumerate(node.input_tensor_names):
            if ctx.get_shape(input_name) != shapeo:
                needs_broadcast_op.append(i)
            else:
                has_correct_shape.append(input_name)
    if needs_broadcast_op:
        has_correct_shape = has_correct_shape[0]
        for i in needs_broadcast_op:
            input_node = node.input_nodes[i]
            # get a tensor with zeros (since there is no Fill op as of opset8)
            sub_node = ctx.MakeNode(
                "Sub",
                [has_correct_shape, has_correct_shape],
                op_name_scope=input_node.name,
            )
            # use add as 'broadcast' op
            add_node = ctx.MakeNode(
                "Add",
                [input_node.output_tensor_names[0], sub_node.output_tensor_names[0]],
                op_name_scope=input_node.name,
            )
            node.input_tensor_names[i] = add_node.output_tensor_names[0]


@flow_op("broadcast_minimum", onnx_op="Min")
@flow_op("broadcast_maximum", onnx_op="Max")
class MinMaxOp:
    @classmethod
    def Version_1(cls, ctx, node, **kwargs):
        shapes = node.output_shapes
        dtypes = node.output_dtypes
        ctx.RemoveNode(node.name)
        _MakeMinOrMaxOp(
            ctx,
            node.op_type,
            node.input_tensor_names,
            node.output_tensor_names,
            shapes,
            dtypes,
        )


class ClipOps:
    @classmethod
    def Version_1(cls, ctx, node, min_val=None, max_val=None, **kwargs):
        # relu6 = min(max(features, 0), 6)
        node.op_type = "Clip"
        if min_val is not None:
            node.attrs["min"] = float(min_val)
        if max_val is not None:
            node.attrs["max"] = float(max_val)

    @classmethod
    def Version_11(cls, ctx, node, min_val=None, max_val=None, **kwargs):
        # add min and max as inputs
        node.op_type = "Clip"
        onnx_dtype = ctx.get_dtype(node.input_tensor_names[0])
        np_dtype = util.ONNX_2_NUMPY_DTYPE[onnx_dtype]
        if min_val is not None:
            clip_min = ctx.MakeConst(
                id_util.UniqueStr("{}_min".format(node.name)),
                np.array(min_val, dtype=np_dtype),
            )
            node.input_tensor_names.append(clip_min.output_tensor_names[0])
        else:
            node.input_tensor_names.append("")
        if max_val is not None:
            clip_max = ctx.MakeConst(
                id_util.UniqueStr("{}_max".format(node.name)),
                np.array(max_val, dtype=np_dtype),
            )
            node.input_tensor_names.append(clip_max.output_tensor_names[0])
        else:
            node.input_tensor_names.append("")


@flow_op(["clip_by_scalar", "clip_by_scalar_min", "clip_by_scalar_max"], onnx_op="Clip")
class ClipByValueOp(ClipOps):
    @classmethod
    def Version_1(cls, ctx, node, **kwargs):
        min_val = node.attrs.get("floating_min", None) or node.attrs.get(
            "integral_min", None
        )
        max_val = node.attrs.get("floating_max", None) or node.attrs.get(
            "integral_max", None
        )
        super().Version_1(ctx, node, min_val, max_val)

    @classmethod
    def Version_11(cls, ctx, node, **kwargs):
        min_val = node.attrs.get("floating_min", None) or node.attrs.get(
            "integral_min", None
        )
        max_val = node.attrs.get("floating_max", None) or node.attrs.get(
            "integral_max", None
        )
        super().Version_11(ctx, node, min_val, max_val)


@flow_op("softmax", "Softmax")
class Softmax:
    @classmethod
    def Version_1(cls, ctx, node, **kwargs):
        # T output = Softmax(T logits). The axis softmax would be performed on is always on -1.
        # T output = Softmax(T input, @int axis). Default axis is 1.
        logits_rank = len(ctx.get_shape(node.input_tensor_names[0]))
        node.attrs["axis"] = logits_rank - 1

    @classmethod
    def Version_11(cls, ctx, node, **kwargs):
        cls.Version_1(ctx, node, **kwargs)


@flow_op("square", None)
class Square:
    @classmethod
    def Version_1(cls, ctx, node, **kwargs):
        node.op_type = "Mul"
        node.input_tensor_names.append(node.input_tensor_names[0])


@flow_op("rsqrt", None)
class Rsqrt:
    @classmethod
    def Version_1(cls, ctx, node, **kwargs):
        node.op_type = "Sqrt"
        op_name = id_util.UniqueStr(node.name)
        reciprocal = ctx.InsertNewNodeOnOutput(
            "Reciprocal", node.output_tensor_names[0], name=op_name
        )
        ctx.CopyShape(node.output_tensor_names[0], reciprocal.output_tensor_names[0])


@flow_op("squared_difference", None)
class SquaredDifference:
    @classmethod
    def Version_1(cls, ctx, node, **kwargs):
        node.op_type = "Sub"
        op_name = id_util.UniqueStr(node.name)
        mul = ctx.InsertNewNodeOnOutput(
            "Mul", node.output_tensor_names[0], name=op_name
        )
        mul.input_tensor_names.append(node.output_tensor_names[0])


@flow_op("sign", onnx_op="Sign")
class Sign:
    @classmethod
    def Version_1(cls, ctx, node, **kwargs):
        """Sign op."""
        # T sign = Sign(T Input)
        node_dtype = ctx.get_dtype(node.output_tensor_names[0])
        util.MakeSure(node_dtype, "Dtype of {} is None".format(node.name))
        if node_dtype in [
            onnx_pb.TensorProto.COMPLEX64,
            onnx_pb.TensorProto.COMPLEX128,
        ]:
            raise ValueError(
                "dtype " + str(node_dtype) + " is not supported in onnx for now"
            )
        zero_name = id_util.UniqueStr("{}_zero".format(node.name))
        ctx.MakeConst(zero_name, np.array(0, dtype=np.float32))
        if node_dtype not in [
            onnx_pb.TensorProto.FLOAT16,
            onnx_pb.TensorProto.FLOAT,
            onnx_pb.TensorProto.DOUBLE,
        ]:
            cast_node_0 = ctx.MakeNode(
                "Cast", [node.input_tensor_names[0]], {"to": onnx_pb.TensorProto.FLOAT}
            )
            greater_node = ctx.MakeNode(
                "Greater", [cast_node_0.output_tensor_names[0], zero_name]
            )
            less_node = ctx.MakeNode(
                "Less", [cast_node_0.output_tensor_names[0], zero_name]
            )
        else:
            greater_node = ctx.MakeNode(
                "Greater", [node.input_tensor_names[0], zero_name]
            )
            less_node = ctx.MakeNode("Less", [node.input_tensor_names[0], zero_name])
        cast_node_1 = ctx.MakeNode(
            "Cast", [greater_node.output_tensor_names[0]], {"to": node_dtype}
        )
        cast_node_2 = ctx.MakeNode(
            "Cast", [less_node.output_tensor_names[0]], {"to": node_dtype}
        )

        shapes = node.output_shapes
        dtypes = node.output_dtypes
        ctx.RemoveNode(node.name)
        ctx.MakeNode(
            "Sub",
            [cast_node_1.output_tensor_names[0], cast_node_2.output_tensor_names[0]],
            outputs=[node.output_tensor_names[0]],
            shapes=shapes,
            dtypes=dtypes,
        )

    @classmethod
    def Version_9(cls, ctx, node, **kwargs):
        node_dtype = ctx.get_dtype(node.output_tensor_names[0])
        util.MakeSure(node_dtype, "dtype of {} is None".format(node.name))
        if node_dtype in [
            onnx_pb.TensorProto.BOOL,
            onnx_pb.TensorProto.COMPLEX64,
            onnx_pb.TensorProto.COMPLEX128,
        ]:
            raise ValueError(
                "dtype " + str(node_dtype) + " is not supported in onnx for now"
            )


@flow_op(["matmul", "batch_matmul"], "MatMul", flow_ibns=["a", "b"])
class MatMul:
    @classmethod
    def Version_1(cls, ctx, node, **kwargs):
        transpose_a = node.attrs.get("transpose_a", 0)
        transpose_b = node.attrs.get("transpose_b", 0)

        if transpose_a != 0:
            shape = ctx.get_shape(node.input_tensor_names[0])
            if shape:
                perm = list(range(0, len(shape)))
                tmp = perm[-1]
                perm[-1] = perm[-2]
                perm[-2] = tmp
                ctx.InsertNewNodeOnInput(
                    node, "Transpose", node.input_tensor_names[0], perm=perm
                )

        if transpose_b != 0:
            shape = ctx.get_shape(node.input_tensor_names[1])
            if shape:
                perm = list(range(0, len(shape)))
                tmp = perm[-1]
                perm[-1] = perm[-2]
                perm[-2] = tmp
                ctx.InsertNewNodeOnInput(
                    node, "Transpose", node.input_tensor_names[1], perm=perm
                )

        unsupported = ["a_is_sparse", "b_is_sparse"]
        for i in unsupported:
            val = node.attrs.get(i, 0)
            if val != 0:
                raise ValueError(node.op_type + " attribute " + i + " is not supported")


@flow_op("erf", onnx_op="Erf")
class Erf:
    @classmethod
    def Version_1(cls, ctx, node, **kwargs):
        """Error function."""
        # constant names
        a1 = "erf_a1"
        a2 = "erf_a2"
        a3 = "erf_a3"
        a4 = "erf_a4"
        a5 = "erf_a5"
        p = "erf_p"
        one = "erf_one"
        null = "erf_null"

        n = node.name
        output_name = node.output_tensor_names[0]
        erf_a1_node = ctx.get_node_by_output("erf_a1")
        if erf_a1_node is None:
            # insert the constants for erf once
            ctx.MakeConst(a1, np.array(0.254829592, dtype=np.float32))
            ctx.MakeConst(a2, np.array(-0.284496736, dtype=np.float32))
            ctx.MakeConst(a3, np.array(1.421413741, dtype=np.float32))
            ctx.MakeConst(a4, np.array(-1.453152027, dtype=np.float32))
            ctx.MakeConst(a5, np.array(1.061405429, dtype=np.float32))
            ctx.MakeConst(p, np.array(0.3275911, dtype=np.float32))
            ctx.MakeConst(one, np.array(1.0, dtype=np.float32))
            ctx.MakeConst(null, np.array(0.0, dtype=np.float32))

        x = node.input_tensor_names[0]

        # erf(x):
        #  sign = 1 if x >= 0 else -1
        #  x = abs(x)
        #  # A&S formula 7.1.26
        #  t = 1.0 / (1.0 + p * x)
        #  y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) *  t * math.exp(-x * x)
        #  return sign * y  # erf(-x) = -erf(x)

        x_node = ctx.MakeNode("Abs", [x], op_name_scope=node.name, name="x")
        negx_node = ctx.MakeNode("Sub", [null, x], op_name_scope=node.name, name="negx")
        is_positive_node = ctx.MakeNode(
            "Greater", [x, null], op_name_scope=node.name, name="isPositive"
        )
        is_positive_value_node = ctx.MakeNode(
            "Cast",
            is_positive_node.output_tensor_names,
            op_name_scope=node.name,
            name="isPositiveValue",
            attr={"to": onnx_pb.TensorProto.FLOAT},
        )
        is_neg_node = ctx.MakeNode(
            "Less", [x, null], op_name_scope=node.name, name="isNeg"
        )
        ig_neg_value_node = ctx.MakeNode(
            "Cast",
            is_neg_node.output_tensor_names,
            op_name_scope=node.name,
            name="isNegValue",
            attr={"to": onnx_pb.TensorProto.FLOAT},
        )
        sign0_node = ctx.MakeNode(
            "Sub",
            [
                is_positive_value_node.output_tensor_names[0],
                ig_neg_value_node.output_tensor_names[0],
            ],
            op_name_scope=node.name,
            name="sign0",
        )
        sign_add_one_node = ctx.MakeNode(
            "Add",
            [sign0_node.output_tensor_names[0], one],
            op_name_scope=node.name,
            name="signAddOne",
        )
        non_zero_node = ctx.MakeNode(
            "Abs",
            sign0_node.output_tensor_names,
            op_name_scope=node.name,
            name="nonZero",
        )
        sign_node = ctx.MakeNode(
            "Sub",
            [
                sign_add_one_node.output_tensor_names[0],
                non_zero_node.output_tensor_names[0],
            ],
            op_name_scope=node.name,
            name="sign",
        )
        num_4_node = ctx.MakeNode(
            "Mul", [x_node.output_tensor_names[0], p], op_name_scope=node.name, name="4"
        )
        num_5_node = ctx.MakeNode(
            "Add",
            [num_4_node.output_tensor_names[0], one],
            op_name_scope=node.name,
            name="5",
        )
        t_node = ctx.MakeNode(
            "Div",
            [one, num_5_node.output_tensor_names[0]],
            op_name_scope=node.name,
            name="t",
        )
        xsq_node = ctx.MakeNode(
            "Mul",
            [x, negx_node.output_tensor_names[0]],
            op_name_scope=node.name,
            name="xsq",
        )
        num_6_node = ctx.MakeNode(
            "Exp", xsq_node.output_tensor_names, op_name_scope=node.name, name="6"
        )
        num_7_node = ctx.MakeNode(
            "Mul",
            [num_6_node.output_tensor_names[0], t_node.output_tensor_names[0]],
            op_name_scope=node.name,
            name="7",
        )
        num_8_node = ctx.MakeNode(
            "Mul",
            [t_node.output_tensor_names[0], a5],
            op_name_scope=node.name,
            name="8",
        )
        num_9_node = ctx.MakeNode(
            "Add",
            [num_8_node.output_tensor_names[0], a4],
            op_name_scope=node.name,
            name="9",
        )
        num_10_node = ctx.MakeNode(
            "Mul",
            [num_9_node.output_tensor_names[0], t_node.output_tensor_names[0]],
            op_name_scope=node.name,
            name="10",
        )
        num_11_node = ctx.MakeNode(
            "Add",
            [num_10_node.output_tensor_names[0], a3],
            op_name_scope=node.name,
            name="11",
        )
        num_12_node = ctx.MakeNode(
            "Mul",
            [num_11_node.output_tensor_names[0], t_node.output_tensor_names[0]],
            op_name_scope=node.name,
            name="12",
        )
        num_13_node = ctx.MakeNode(
            "Add",
            [num_12_node.output_tensor_names[0], a2],
            op_name_scope=node.name,
            name="13",
        )
        num_14_node = ctx.MakeNode(
            "Mul",
            [num_13_node.output_tensor_names[0], t_node.output_tensor_names[0]],
            op_name_scope=node.name,
            name="14",
        )
        num_15_node = ctx.MakeNode(
            "Add",
            [num_14_node.output_tensor_names[0], a1],
            op_name_scope=node.name,
            name="15",
        )
        num_16_node = ctx.MakeNode(
            "Mul",
            [num_15_node.output_tensor_names[0], num_7_node.output_tensor_names[0]],
            op_name_scope=node.name,
            name="16",
        )
        num_17_node = ctx.MakeNode(
            "Sub",
            [one, num_16_node.output_tensor_names[0]],
            op_name_scope=node.name,
            name="17",
        )

        shapes = node.output_shapes
        dtypes = node.output_dtypes
        ctx.RemoveNode(node.name)
        ctx.MakeNode(
            "Mul",
            [num_17_node.output_tensor_names[0], sign_node.output_tensor_names[0]],
            outputs=[output_name],
            name=n,
            shapes=shapes,
            dtypes=dtypes,
        )

    @classmethod
    def Version_9(cls, ctx, node, **kwargs):
        pass


@flow_op("broadcast_floor_mod", onnx_op="FloorMod")
class FloorMod:
    @classmethod
    def Version_7(cls, ctx, node, **kwargs):
        # T output = FloorMod(T x, T y)
        div = ctx.MakeNode(op_type="Div", inputs=node.input_tensor_names)
        dtype = ctx.get_dtype(node.input_tensor_names[0])
        if dtype in [
            onnx_pb.TensorProto.FLOAT,
            onnx_pb.TensorProto.FLOAT16,
            onnx_pb.TensorProto.DOUBLE,
        ]:
            div = ctx.MakeNode(op_type="Floor", inputs=div.output_tensor_names)

        mul = ctx.MakeNode(
            op_type="Mul",
            inputs=[div.output_tensor_names[0], node.input_tensor_names[1]],
        )
        # res node will take over shape&dtype&output connection info of original "node"
        shapes = node.output_shapes
        dtypes = node.output_dtypes
        ctx.RemoveNode(node.name)
        ctx.MakeNode(
            op_type="Sub",
            inputs=[node.input_tensor_names[0], mul.output_tensor_names[0]],
            name=node.name,
            outputs=node.output_tensor_names,
            shapes=shapes,
            dtypes=dtypes,
        )


@flow_op("round", onnx_op="Round")
class Round:
    @classmethod
    def Version_11(cls, ctx, node, **kwargs):
        pass


def _AddCastToInputs(graph, node, supported_dtypes, target_dtype):
    is_support = True
    for inp in node.input_tensor_names:
        if graph.get_dtype(inp) not in supported_dtypes:
            is_support = False
            break
    if not is_support:
        for inp in node.input_tensor_names:
            inp_cast = graph.InsertNewNodeOnInput(node, "Cast", inp, to=target_dtype)
            graph.CopyShape(inp, inp_cast.output_tensor_names[0])
            graph.set_dtype(inp_cast.output_tensor_names[0], target_dtype)


def _AddCastToOutput(graph, node):
    # oneflow logical ops produce int8 tensor while onnx logical ops produce bool tensor
    output = node.output_tensor_names[0]
    cast_node = graph.InsertNewNodeOnOutput(
        "Cast", output, id_util.UniqueStr("cast"), to=graph.get_dtype(output)
    )
    graph.CopyShape(output, node.output_tensor_names[0])
    graph.set_dtype(node.output_tensor_names[0], TensorProto.BOOL)


# oneflow doesn't have logical_not and broadcast_logical_or, but
# it is easy to implement onnx converter in advance
@flow_op("logical_not", onnx_op="Not")
class DirectOp:
    @classmethod
    def Version_1(cls, ctx, node, **kwargs):
        _AddCastToOutput(ctx, node)


@flow_op("broadcast_logical_and", onnx_op="And", flow_ibns=["x", "y"])
@flow_op("broadcast_logical_or", onnx_op="Or", flow_ibns=["x", "y"])
class BroadcastOp(common.BroadcastOp):
    @classmethod
    def Version_1(cls, ctx, node, **kwargs):
        _AddCastToOutput(ctx, node)
        super().Version_1(ctx, node, **kwargs)


@flow_op(
    ["broadcast_equal", "broadcast_not_equal"],
    ["Equal", "NotEqual"],
    flow_ibns=["x", "y"],
)
class Equal:
    @classmethod
    def Version_7(cls, ctx, node, **kwargs):
        # T2 output = Equal(T1, x, T1 y), T1 \in {bool, int32, int64}
        _AddCastToOutput(ctx, node)
        need_not = node.op_type == "NotEqual"
        supported_dtypes = [TensorProto.BOOL, TensorProto.INT32, TensorProto.INT64]
        if any(
            [
                ctx.get_dtype(inp) not in supported_dtypes
                for inp in node.input_tensor_names
            ]
        ):
            raise ValueError(
                "Version 7 Equal op only supports bool, int32 and int64 inputs. Please set opset > 11 and try again."
            )
        if need_not:
            node.op_type = "Equal"
            output_name = node.output_tensor_names[0]
            not_node = ctx.InsertNewNodeOnOutput(
                "Not", output_name, name=id_util.UniqueStr(node.name)
            )
            ctx.CopyShape(output_name, not_node.output_tensor_names[0])
            ctx.CopyDtype(output_name, not_node.output_tensor_names[0])

    @classmethod
    def Version_11(cls, ctx, node, **kwargs):
        # starting with opset-11, equal supports all types
        _AddCastToOutput(ctx, node)
        need_not = node.op_type == "NotEqual"
        if need_not:
            node.op_type = "Equal"
            output_name = node.output_tensor_names[0]
            not_node = ctx.InsertNewNodeOnOutput(
                "Not", output_name, name=id_util.UniqueStr(node.name)
            )
            ctx.CopyShape(output_name, not_node.output_tensor_names[0])
            ctx.CopyDtype(output_name, not_node.output_tensor_names[0])


@flow_op(
    ["broadcast_greater", "broadcast_less"], ["Greater", "Less"], flow_ibns=["x", "y"]
)
class GreaterLess:
    @classmethod
    def Version_7(cls, ctx, node, **kwargs):
        _AddCastToOutput(ctx, node)
        # T2 output = Greater(T1 x, T1 y), T2=tensor(bool)
        # T2 output = Less(T1 x, T1 y), T2=tensor(bool)
        # Great/Less in opset7 only supports limited types, insert Cast if needed
        supported_dtypes = [TensorProto.FLOAT, TensorProto.FLOAT16, TensorProto.DOUBLE]
        target_dtype = TensorProto.FLOAT
        _AddCastToInputs(ctx, node, supported_dtypes, target_dtype)


@flow_op("broadcast_greater_equal", onnx_op="Less", flow_ibns=["x", "y"])
@flow_op("broadcast_less_equal", onnx_op="Greater", flow_ibns=["x", "y"])
class GreaterLessEqual:
    @classmethod
    def Version_7(cls, ctx, node, **kwargs):
        _AddCastToOutput(ctx, node)
        GreaterLess.Version_7(ctx, node, **kwargs)
        output_name = node.output_tensor_names[0]
        new_node = ctx.InsertNewNodeOnOutput(
            "Not", output_name, name=id_util.UniqueStr(node.name)
        )
        ctx.CopyShape(output_name, new_node.output_tensor_names[0])
        ctx.set_dtype(new_node.output_tensor_names[0], ctx.get_dtype(output_name))
