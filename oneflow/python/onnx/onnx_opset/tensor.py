# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""
tensor
"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import sys

import numpy as np
from onnx import numpy_helper
from onnx import onnx_pb
from onnx.onnx_pb import TensorProto

import oneflow.python.onnx
from oneflow.python.framework import id_util
from oneflow.python.onnx import constants, util
from oneflow.python.onnx.graph_builder import GraphBuilder
from oneflow.python.onnx.handler import flow_op
from oneflow.python.onnx.onnx_opset import nn, math

logger = logging.getLogger(__name__)


# pylint: disable=unused-argument,missing-docstring,unused-variable,pointless-string-statement


def _convert_shapenode_to_int64(ctx, node, input_number):
    """cast int32 shape into int64 shape."""
    name = node.input[input_number]

    cast_node = ctx.insert_new_node_on_input(node, "Cast", name)
    cast_node.set_attr("to", onnx_pb.TensorProto.INT64)
    ctx.set_dtype(cast_node.output[0], onnx_pb.TensorProto.INT64)
    ctx.copy_shape(name, cast_node.output[0])


def _wrap_concat_with_cast(ctx, node):
    """wrap concat in casts for opset < 8 since it only supports."""
    supported_types = [onnx_pb.TensorProto.FLOAT, onnx_pb.TensorProto.FLOAT16]
    dtype = ctx.get_dtype(node.output[0])
    need_casting = dtype not in supported_types
    if need_casting:
        output_name = node.output[0]
        # cast each inputs to float
        for i, inp in enumerate(node.inputs):
            input_cast = ctx.insert_new_node_on_input(node, "Cast", node.input[i])
            input_cast.set_attr("to", onnx_pb.TensorProto.FLOAT)
            ctx.set_dtype(input_cast.output[0], onnx_pb.TensorProto.FLOAT)
        next_nodes = ctx.find_output_consumers(node.output[0])
        # cast output back to dtype unless the next op is a cast
        if next_nodes[0].type != "Cast":
            op_name = id_util.UniqueStr(node.name)
            output_cast = ctx.insert_new_node_on_output(
                "Cast", output_name, name=op_name
            )
            output_cast.set_attr("to", dtype)
            ctx.set_dtype(output_cast.output[0], dtype)
            ctx.copy_shape(output_name, output_cast.output[0])


@flow_op("reshape", "Reshape")
class Reshape:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        # T output = Reshape(T tensor, Tshape shape, @type Tshape)
        # T reshaped = Reshape(T data, @INTS shape) - but takes a optional 2nd input for shape
        return
        shape_node = node.inputs[1]
        shape = shape_node.get_tensor_value()
        if shape is None:
            logger.error("Reshape on node %s does not have a const shape", node.name)
            return
        ctx.remove_input(node, node.input[1])
        node.set_attr("shape", shape)
        ctx.set_shape(node.output[0], shape)

    @classmethod
    def version_5(cls, ctx, node, **kwargs):
        dtype = ctx.get_dtype(node.output[0])
        need_casting = dtype in [
            onnx_pb.TensorProto.INT32,
            onnx_pb.TensorProto.INT16,
            onnx_pb.TensorProto.INT64,
        ]
        shape_node = ctx.make_const(
            id_util.UniqueStr("shape"), np.array(node.get_attr_value("shape"))
        )
        node.input = node.input + [shape_node.name]
        if ctx.opset >= 8 or not need_casting:
            # onnx reshape can handle the type - done
            return

        # onnx < opset 8 does not know reshape for other types than float*, wrap the reshape in casts
        input_cast = ctx.insert_new_node_on_input(node, "Cast", node.input[0])
        input_cast.set_attr("to", onnx_pb.TensorProto.FLOAT)
        ctx.copy_shape(node.output[0], input_cast.output[0])

        # if the next node is already a cast we don't need to insert another one
        next_nodes = ctx.find_output_consumers(node.output[0])
        if len(next_nodes) != 1 or next_nodes[0].type != "Cast":
            op_name = id_util.UniqueStr(node.name)
            output_cast = ctx.insert_new_node_on_output(
                "Cast", node.output[0], name=op_name
            )
            output_cast.set_attr("to", dtype)
            ctx.set_dtype(output_cast.output[0], dtype)
            ctx.copy_shape(node.output[0], output_cast.output[0])


@flow_op("squeeze", "Squeeze")
class Squeeze:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        # T output = Squeeze(T input, @list(int) squeeze_dims)
        # T squeezed = Squeeze(T data, @AttrType.INTS axes), axes are list of positive integers.
        axis = node.get_attr_value("axes")

        neg_axis = any([val < 0 for val in axis])
        if neg_axis:
            shape = ctx.get_shape(node.input[0])
            util.make_sure(shape is not None, "squeeze input shape cannot be None")
            shape_len = len(shape)
            axis = [a + shape_len if a < 0 else a for a in axis]
        node.set_attr("axes", axis)

    @classmethod
    def version_11(cls, ctx, node, **kwargs):
        # Opset 11 supports negative axis, but core logic is same
        cls.version_1(ctx, node, **kwargs)


@flow_op("transpose", onnx_op="Transpose")
class Transpose:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        # T y = Transpose(T x, Tperm perm, @type Tperm)
        # T transposed = Transpose(T data, @INTS perm)
        if len(node.input) > 1:
            perm = node.inputs[1]
            if perm.is_const():
                # perms is passed as const
                dims = perm.get_tensor_value()
                ctx.remove_input(node, node.input[1])
                node.set_attr("perm", dims)
            else:
                util.make_sure(False, "perm can't be dynamic in ONNX")
        else:
            # graph rewrite moved perm to attribute
            pass


@flow_op("concat", "Concat")
class Concat:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        # old concat op has axis as input[0]
        axis_val = node.get_attr_value("axis")

        if axis_val < 0:
            input_shape = ctx.get_shape(node.input[0])
            axis_val = len(input_shape) + axis_val
        node.set_attr("axis", axis_val)

        if ctx.opset < 8:
            # opset < 8: might need to wrap concat in casts since only float is supported
            _wrap_concat_with_cast(ctx, node)
            return

    @classmethod
    def version_11(cls, ctx, node, **kwargs):
        # Opset 11 supports negative axis, but core logic is same
        cls.version_1(ctx, node, **kwargs)


def _make_gathernd_inner_loop(ctx, params, index, dtype):
    """create the inner loop for GatherNd."""
    # gather_cur = params
    # for (int i = 0; i < size(index); i++)
    #   gather_res = gather(gather_cur, index[i])
    scope_name = id_util.UniqueStr("gathernd_inner_loop")
    trip_node = ctx.make_node("Size", [index.output[0]])
    cond_const = ctx.make_const(id_util.UniqueStr("cond"), np.ones((), dtype=np.bool))
    trip_name = id_util.UniqueStr("i")
    cond_name = id_util.UniqueStr("cond")
    cond_out_name = id_util.UniqueStr("cond_out")
    cur_name = id_util.UniqueStr("gather_cur")
    result_name = id_util.UniqueStr("res")

    # body graph creation
    g = ctx.create_new_graph_with_same_config()
    g.add_graph_input(trip_name, TensorProto.INT64, [1])
    g.add_graph_input(cond_name, TensorProto.BOOL, [])
    g.add_graph_input(cur_name, dtype, [])
    g.parent_graph = ctx

    index_i = g.make_node("Gather", [index.output[0], trip_name], attr={"axis": 0})
    gather = g.make_node("Gather", [cur_name, index_i.output[0]], attr={"axis": 0})
    g.make_node(
        "Squeeze", [gather.output[0]], attr={"axes": [0]}, outputs=[result_name]
    )
    g.make_node("Identity", [cond_name], outputs=[cond_out_name])

    g.add_graph_output(cond_out_name, TensorProto.BOOL, [])
    g.add_graph_output(result_name, dtype, [])

    inner_loop = ctx.make_node(
        "Loop",
        [trip_node.output[0], cond_const.output[0], params],
        op_name_scope=scope_name,
        skip_conversion=False,
    )
    inner_loop.set_body_graph_as_attr("body", g)
    return inner_loop


def make_gathernd(ctx, params, indices, output, scope_name, t_params, shapes, dtypes):
    """make GatherNd op."""
    # Tparams output = GatherNd(Tparams params, Tidx indices)
    scope_name = id_util.UniqueStr(scope_name)
    # reshape indices into [sum(indices[:-1]), indices[-1]]
    indices_shape = ctx.make_node("Shape", [indices], dtypes=[TensorProto.INT64])
    indices_size = ctx.make_node("Size", [indices])
    attr = {"axes": [0], "ends": [sys.maxsize], "starts": [-1]}
    inputs_map = {"data": indices_shape.output[0], **attr}
    inner_shape = GraphBuilder(ctx).make_slice(inputs_map, dtypes=[TensorProto.INT64])
    outter_shape = ctx.make_node(
        "Div", [indices_size.output[0], inner_shape], dtypes=[TensorProto.INT64]
    )
    flatten_shape = ctx.make_node(
        "Concat",
        [outter_shape.output[0], inner_shape],
        attr={"axis": 0},
        dtypes=[TensorProto.INT64],
    )
    flatten_indices = ctx.make_node("Reshape", [indices, flatten_shape.output[0]])

    # outter loop for each index
    # for (int i=0; i<outter_shape; i++) inner_loop(params, flatten_indices[i])
    cond_const = ctx.make_const(id_util.UniqueStr("cond"), np.ones((), dtype=np.bool))
    dummy_const = ctx.make_const(id_util.UniqueStr("dummy"), np.ones((), dtype=np.int64))

    # body graph creation
    g = ctx.create_new_graph_with_same_config()
    trip_name = id_util.UniqueStr("i")
    cond_name = id_util.UniqueStr("cond")
    cond_out_name = id_util.UniqueStr("cond_out")
    dummy_name = id_util.UniqueStr("dummy")
    dummy_out_name = id_util.UniqueStr("dummy_out")
    result_name = id_util.UniqueStr("res")

    g.add_graph_input(trip_name, TensorProto.INT64, [1])
    g.add_graph_input(cond_name, TensorProto.BOOL, [])
    g.add_graph_input(dummy_name, t_params, [])
    g.parent_graph = ctx

    index = g.make_node(
        "Gather", [flatten_indices.output[0], trip_name], attr={"axis": 0}
    )
    index_squeeze = g.make_node("Squeeze", [index.output[0]], attr={"axes": [0]})
    # inner loop to gather result
    inner_loop = _make_gathernd_inner_loop(g, params, index_squeeze, t_params)
    g.make_node("Identity", [cond_name], outputs=[cond_out_name])
    g.make_node("Identity", [dummy_name], outputs=[dummy_out_name])
    g.make_node("Identity", [inner_loop.output[0]], outputs=[result_name])

    g.add_graph_output(cond_out_name, TensorProto.BOOL, [])
    g.add_graph_output(dummy_out_name, t_params, [])
    g.add_graph_output(result_name, t_params, [])

    gathernd_loop = ctx.make_node(
        "Loop",
        [outter_shape.output[0], cond_const.output[0], params],
        output_count=2,
        op_name_scope=scope_name,
        skip_conversion=False,
    )
    gathernd_loop.set_body_graph_as_attr("body", g)

    # reshape to target shape
    # output shape of gathernd: indices.shape[:-1] + gathernd_output.shape[1:]
    inner_loop_shape = ctx.make_node(
        "Shape", [gathernd_loop.output[1]], dtypes=[TensorProto.INT64]
    )
    # workaround in case gathernd_loop is 1-dimensional
    one_const = ctx.make_const(id_util.UniqueStr("one"), np.array([1], dtype=np.int64))
    inner_loop_shape_ = ctx.make_node(
        "Concat",
        [inner_loop_shape.output[0], one_const.output[0]],
        attr={"axis": 0},
        dtypes=[TensorProto.INT64],
    )
    attr = {"axes": [0], "ends": [sys.maxsize], "starts": [1]}
    inputs_map = {"data": inner_loop_shape_.output[0], **attr}
    output_inner_shape = GraphBuilder(ctx).make_slice(
        inputs_map, dtypes=[TensorProto.INT64]
    )
    attr = {"axes": [0], "ends": [-1], "starts": [0]}
    inputs_map = {"data": indices_shape.output[0], **attr}
    indices_outter_shape = GraphBuilder(ctx).make_slice(
        inputs_map, dtypes=[TensorProto.INT64]
    )
    output_shape_ = ctx.make_node(
        "Concat",
        [indices_outter_shape, output_inner_shape],
        attr={"axis": 0},
        dtypes=[TensorProto.INT64],
    )
    attr = {"axes": [0], "ends": [-1], "starts": [0]}
    inputs_map = {"data": output_shape_.output[0], **attr}
    output_shape = GraphBuilder(ctx).make_slice(inputs_map, dtypes=[TensorProto.INT64])
    ctx.make_node(
        "Reshape",
        [gathernd_loop.output[1], output_shape],
        outputs=[output],
        shapes=shapes,
        dtypes=dtypes,
    )


@flow_op("gather_nd", onnx_op="GatherND", flow_inputs=["params", "indices"])
class GatherND:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        # Tparams output = GatherNd(Tparams params, Tidx indices)
        params = node.input[0]
        indices = node.input[1]
        output = node.output[0]
        # same as the attr Tparams
        t_params = ctx.get_dtype(params)
        util.make_sure(t_params, "Dtype of {} is None".format(indices))
        shapes = node.output_shapes
        dtypes = node.output_dtypes
        ctx.remove_node(node.name)
        make_gathernd(ctx, params, indices, output, node.name, t_params, shapes, dtypes)

    @classmethod
    def version_11(cls, ctx, node, **kwargs):
        # indicies input
        input1 = node.input[1]
        target_dtype = TensorProto.INT64
        if ctx.get_dtype(input1) != TensorProto.INT64:
            inp_cast = ctx.insert_new_node_on_input(
                node, "Cast", input1, to=target_dtype
            )
            ctx.copy_shape(input1, inp_cast.output[0])
            ctx.set_dtype(inp_cast.output[0], target_dtype)


@flow_op("cast", "Cast")
class Cast:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        # DstT y = Cast(SrcT x, @type SrcT, @type DstT)
        # T2 output = Cast(T1 input, @STRING to)
        dst = node.get_attr("dtype")
        dst = oneflow.python.onnx.util.ONNX_DTYPE_NAMES[dst]
        node.set_attr("to", dst)

    @classmethod
    def version_6(cls, ctx, node, **kwargs):
        dst = node.get_attr_value("dtype")
        node.set_attr("to", dst)

    @classmethod
    def version_9(cls, ctx, node, **kwargs):
        cls.version_6(ctx, node, **kwargs)


@flow_op("identity", "Identity")
class Identity:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        pass
