# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""
controlflow
"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

import numpy as np

from onnx.onnx_pb import TensorProto
from oneflow.python.onnx import utils
from oneflow.python.onnx.handler import flow_op
from oneflow.python.onnx.utils import make_sure

logger = logging.getLogger(__name__)


# pylint: disable=unused-argument,missing-docstring

def get_inputs_for_current_iteration(g, input_id, iter_index):
    cond_gather_node = g.make_node("Gather", [input_id, iter_index])
    cur_cond_val_scalar_node = g.make_node("Squeeze", [cond_gather_node.output[0]], attr={"axes": [0]})
    return cur_cond_val_scalar_node.output[0]


def create_loop_body_graph(parent_g, gather_input_ids, output_data_type, output_shape, trip_count_input_ids,
                           rank, loop_name):
    g = parent_g.create_new_graph_with_same_config()
    g.parent_graph = parent_g
    iter_name = utils.make_name("i")
    cond_name = utils.make_name("cond")
    fake_var_name = utils.make_name("fake_var")

    g.add_graph_input(iter_name, TensorProto.INT64, (1,))  # iteration_num
    g.add_graph_input(cond_name, TensorProto.BOOL, ())  # condition
    g.add_graph_input(fake_var_name, TensorProto.FLOAT, ())  # loop-carried dependency

    # get the i'th value of condition
    cond_input_id = gather_input_ids[0]
    cond_input_id_for_current_iter = get_inputs_for_current_iteration(g, cond_input_id, iter_name)

    # get the i'th value of true values
    true_input_id = gather_input_ids[1]
    true_input_id_for_current_iter = get_inputs_for_current_iteration(g, true_input_id, iter_name)

    # get the i'th value of false values
    false_input_id = gather_input_ids[2]
    false_input_id_for_current_iter = get_inputs_for_current_iteration(g, false_input_id, iter_name)

    input_ids_for_current_iter = [cond_input_id_for_current_iter, true_input_id_for_current_iter,
                                  false_input_id_for_current_iter]
    output_id = None
    rank -= 1
    if rank >= 1:
        loop_1 = create_loop_op(g, input_ids_for_current_iter, output_data_type, output_shape[1:],
                                trip_count_input_ids, rank)
        output_id = loop_1.output[1]
    elif rank == 0:
        _, if_node_output_id = create_if_op(g, input_ids_for_current_iter, output_data_type, output_shape[1:])
        output_id = if_node_output_id

    output_identity_name = utils.make_name("loop_output")
    loop_output_id = utils.port_name(output_identity_name)
    g.make_node(
        'Identity',
        [output_id],
        outputs=[loop_output_id],
        name=output_identity_name
    )

    cond_identity_name = utils.make_name("cond_output")
    cond_output_id = utils.port_name(cond_identity_name)
    g.make_node(
        'Identity',
        [cond_name],
        outputs=[cond_output_id],
        name=cond_identity_name
    )

    fake_var_identity_name = utils.make_name("fake_var_output")
    fake_var_output_id = utils.port_name(fake_var_identity_name)
    g.make_node(
        'Identity',
        [fake_var_name],
        outputs=[fake_var_output_id],
        name=fake_var_identity_name
    )

    g.add_graph_output(cond_output_id, TensorProto.BOOL, ())
    g.add_graph_output(fake_var_output_id, TensorProto.FLOAT, ())

    # use None for all dims, just keep original rank. Because it is observed, dims might be changed in loop.
    g.add_graph_output(loop_output_id, output_data_type, utils.create_vague_shape_like(output_shape[1:]))

    return g


def create_if_op(g, input_ids, output_data_type, output_shape):
    op_name = utils.make_name("If")
    true_graph = create_body_graph_for_if_branch(g, output_data_type, output_shape, input_ids[1], op_name)
    false_graph = create_body_graph_for_if_branch(g, output_data_type, output_shape, input_ids[2], op_name)
    out_name = utils.port_name(op_name)

    # output a scalar
    if_node = g.make_node("If", [input_ids[0]], outputs=[out_name], name=op_name, skip_conversion=False)
    if_node.set_body_graph_as_attr("then_branch", true_graph)
    if_node.set_body_graph_as_attr("else_branch", false_graph)
    return if_node, out_name


def create_body_graph_for_if_branch(parent_g, data_type, output_shape, chosen_cur_cond_val_out_name, op_name):
    g = parent_g.create_new_graph_with_same_config()
    g.parent_graph = parent_g
    name = utils.make_name("Identity")
    g.make_node(
        'Identity',
        inputs=[chosen_cur_cond_val_out_name],
        outputs=['y'],
        name=name
    )
    g.add_graph_output("y", data_type, utils.create_vague_shape_like(output_shape))
    return g


# gather_input_ids is 1-D tensor, containing 3 elements:
# 0: condition data to gather on
# 1: true result to gather on
# 2: false result to father on
def create_loop_op(g, gather_input_ids, output_type, output_shape, trip_count_input_ids, rank):
    cond_var_name = utils.make_name("cond_var")
    g.make_const(cond_var_name, np.array(True, dtype=np.bool))

    # Loop requires at least a variable, add a useless fake variable.
    fake_val_name = utils.make_name("fake_var")
    g.make_const(fake_val_name, np.array(0.0, dtype=np.float32))

    if rank < 1:
        raise ValueError("rank is < 1")
    trip_count_input_id = trip_count_input_ids[-1 * rank]

    loop_inputs = [trip_count_input_id,  # trip count
                   cond_var_name,  # termination condition
                   fake_val_name  # initial value of loop-carried dependencies
                   ]
    # define an extra scan output
    loop_node = g.make_node("Loop", loop_inputs, output_count=2, op_name_scope="select_loop",
                            skip_conversion=False)
    loop_body = create_loop_body_graph(g, gather_input_ids, output_type, output_shape, trip_count_input_ids,
                                       rank, loop_node.name)
    loop_node.set_body_graph_as_attr("body", loop_body)
    return loop_node


def make_range_const(ctx, start, limit, delta, output, scope_name, shape, dtype):
    """make Range subgraph if all inputs are const."""
    # T range = Range(T start, T limit, T delta)
    # V v_final_and_scan_outputs = Loop(int64 M, B cond, V v_initial)
    base_name = utils.make_name(scope_name)
    start = ctx.get_node_by_output(start).get_tensor_value(as_list=False)
    limit = ctx.get_node_by_output(limit).get_tensor_value(as_list=False)
    delta = ctx.get_node_by_output(delta).get_tensor_value(as_list=False)
    val = np.arange(start, limit, delta, dtype=start.dtype)
    const_range = ctx.make_const(base_name, val)
    ctx.make_node("Identity", [const_range.output[0]], shapes=[shape], dtypes=[dtype], outputs=[output])


def make_range_non_const(ctx, start, limit, delta, output, scope_name, shape, dtype):
    """make Range subgraph."""
    # T range = Range(T start, T limit, T delta)
    # V v_final_and_scan_outputs = Loop(int64 M, B cond, V v_initial)
    base_name = utils.make_name(scope_name)

    # trip_count
    diff_node = ctx.make_node("Sub",
                              [limit, start],
                              op_name_scope=base_name,
                              name=utils.make_name("diff"))
    diff_output = diff_node.output[0]

    delta_cast = delta
    if dtype in [TensorProto.INT32, TensorProto.INT64]:
        cast_node = ctx.make_node("Cast", [diff_output], op_name_scope=base_name,
                                  name="cast_diff", attr={"to": TensorProto.FLOAT})
        diff_output = cast_node.output[0]

        cast_node = ctx.make_node("Cast", [delta], op_name_scope=base_name, name="cast_delta",
                                  attr={"to": TensorProto.FLOAT})
        delta_cast = cast_node.output[0]
    div_node = ctx.make_node("Div", [diff_output, delta_cast], op_name_scope=base_name, name="div")
    ceil_node = ctx.make_node("Ceil", [div_node.output[0]], op_name_scope=base_name, name="ceil")
    trip_count_node = ctx.make_node("Cast", [ceil_node.output[0]], op_name_scope=base_name, name="trip_cnt",
                                    attr={"to": TensorProto.INT64})

    # cond
    # Use initializer here since Constant OP before opset 9 does not support bool type
    cond_name = "{}_cond".format(base_name)
    ctx.make_const(cond_name, np.ones((), dtype=bool))

    # body
    g = ctx.create_new_graph_with_same_config()
    g.parent_graph = ctx
    g.add_graph_input("i", TensorProto.INT64, [])
    g.add_graph_input("cond", TensorProto.BOOL, [])
    g.add_graph_input("prev", dtype, [])

    g.make_node("Identity", ["cond"], outputs=["cond_out"])
    g.make_node("Add", ["prev", delta], outputs=["current"], name=utils.make_name("add"))
    g.make_node("Identity", ["prev"], outputs=["range"])

    g.add_graph_output("cond_out", TensorProto.BOOL, [])
    g.add_graph_output("current", dtype, [])
    g.add_graph_output("range", dtype, [])

    # loop
    loop_inputs = [trip_count_node.output[0], cond_name, start]
    loop_node = ctx.make_node("Loop", loop_inputs, output_count=2, op_name_scope=base_name, name="loop")
    loop_node.set_body_graph_as_attr("body", g)

    ctx.make_node("Identity", [loop_node.output[1]], name=base_name, shapes=[shape], dtypes=[dtype], outputs=[output])


def make_range(ctx, start, limit, delta, output, scope_name, shape, dtype):
    if all(ctx.get_node_by_output(n).is_const() for n in [start, limit, delta]) is True:
        make_range_const(ctx, start, limit, delta, output, scope_name, shape, dtype)
    else:
        make_range_non_const(ctx, start, limit, delta, output, scope_name, shape, dtype)


@flow_op(["If", "Loop", "Scan"])
class PassThroughOp:
    @classmethod
    def version_7(cls, ctx, node, **kwargs):
        pass

    @classmethod
    def version_11(cls, ctx, node, **kwargs):
        # no change needed
        # loop has 1 less mandatory input
        # if = only doc changes
        # scan has 1 less mandatory input and 4 extra attrs
        pass


@flow_op("Range")
class Range:
    @classmethod
    def version_7(cls, ctx, node, **kwargs):
        """Range."""
        # T range = Range(T start, T limit, T delta)
        # V v_final_and_scan_outputs = Loop(int64 M, B cond, V v_initial)
        dtype = node.get_attr_int("Tidx")
        shape = node.output_shapes[0]
        utils.make_sure(dtype is not None, "Tidx of %s is None", node.name)
        ctx.remove_node(node.name)
        make_range(ctx, node.input[0], node.input[1], node.input[2],
                   node.output[0], node.name, shape, dtype)

    @classmethod
    def version_11(cls, ctx, node, **kwargs):
        # opset 11 implements Range op explicitly
        pass


@flow_op("Select")
class Select:
    @classmethod
    def version_8(cls, ctx, node, **kwargs):
        # T output = Select(bool condition, T x, T y)
        # V v_final_and_scan_outputs = Loop(int64 M, B cond, V v_initial)
        utils.make_sure(len(node.input) > 1, "Select with only condition is not supported.")

        true_data_type = ctx.get_dtype(node.input[1])
        true_data_shape = ctx.get_shape(node.input[1])
        make_sure(true_data_type is not None, "select true data dtype cannot be None")
        make_sure(true_data_shape is not None, "select true data shape cannot be None")

        condition_shape = ctx.get_shape(node.input[0])
        utils.make_sure(condition_shape is not None, "Shape of {} is None".format(node.input[0]))
        rank = len(condition_shape)

        utils.make_sure(rank >= 0, "rank should be >= 0")
        val_output_id = None
        if rank > 0:
            # create nodes getting shape of condition
            shape_node_output_shape = [rank]
            shape_node = ctx.make_node("Shape", [node.input[0]], op_name_scope=node.name,
                                       shapes=[shape_node_output_shape], dtypes=[TensorProto.INT64])

            # todo(pengwa), move those leveraging rewrite_incomplete_type_support_onnxruntime after shape inferencing
            # bug is fixed.
            # workaround: onnxruntime does not support Split-2, add cases before and after.
            target_dtype = TensorProto.FLOAT
            shape_f_node = ctx.make_node("Cast", [shape_node.output[0]], attr={"to": target_dtype},
                                         shapes=[shape_node_output_shape], dtypes=[target_dtype],
                                         op_name_scope=node.name)

            split_attr = [1 for i in range(rank)]
            output_shapes = [[1] for i in range(rank)]
            output_dtypes = [target_dtype for i in range(rank)]
            split_node = ctx.make_node("Split", [shape_f_node.output[0]], output_count=rank,
                                       attr={"split": split_attr}, shapes=output_shapes,
                                       dtypes=output_dtypes, op_name_scope=node.name)

            trip_cnts = []
            for i in range(rank):
                output_id = split_node.output[i]
                output_shape = ctx.get_shape(output_id)
                target_dtype = TensorProto.INT64
                shape_i_node = ctx.make_node("Cast", [output_id], attr={"to": target_dtype},
                                             shapes=[output_shape], dtypes=[target_dtype],
                                             op_name_scope=node.name)
                trip_cnts.append(shape_i_node.output[0])
            # workaround ends

            loop_node = create_loop_op(ctx, node.input, true_data_type, true_data_shape, trip_cnts, rank)

            val_output_id = loop_node.output[1]
        elif rank == 0:
            _, val_output_id = create_if_op(ctx, node.input, true_data_type, true_data_shape)

        ctx.copy_shape(node.output[0], val_output_id)
        ctx.set_dtype(node.output[0], true_data_type)
        ctx.remove_node(node.name)
        ctx.make_node("Identity", [val_output_id], outputs=node.output,
                      shapes=[ctx.get_shape(val_output_id)], dtypes=[true_data_type])

    @classmethod
    def version_9(cls, ctx, node, **kwargs):
        # T output = Select(bool condition, T x, T y)
        # T1 output = Where(bool condition, T1 x, T1 y)
        # NOTE: condition can be 1-dimension in tensorflow, while in onnx,
        # it should be broadcastable with other two inputs
        node.type = "Where"
        cond_shape = ctx.get_shape(node.input[0])
        make_sure(cond_shape is not None, "shape of {} is None".format(node.input[0]))
        input_shape = ctx.get_shape(node.input[1])
        if input_shape is None:
            input_shape = ctx.get_shape(node.input[2])
        make_sure(input_shape is not None, "input shape of {} is None".format(node.name))
        input_rank = len(input_shape)
        # if cond shape is 1-dimensional while input has higher rank, need to be reshaped to broadcast
        if len(cond_shape) == 1 and input_rank > 1:
            broadcast_shape = [cond_shape[0]] + [1] * (input_rank - 1)
            shape_const = ctx.make_const(utils.make_name(node.name), np.array(broadcast_shape, dtype=np.int64))
            reshape = ctx.make_node("Reshape", [node.input[0], shape_const.output[0]])
            ctx.replace_input(node, node.input[0], reshape.output[0])


@flow_op("Where")
class Where:
    @classmethod
    def version_9(cls, ctx, node, **kwargs):
        # T_y output = Where(T_x condition), return indices of elements whose value are True
        node.type = "NonZero"
        # in onnx, indices are returned in this way [[ind_a_0, ind_b_0, ...], [ind_a_1, ind_b_1,...]];
        # while in tf, the result will be [[ind_a_0, ind_a_1, ...], [ind_b_0, ind_b_1, ...], ...]
        # this is the reason a transpose node inserted here.
        transpose_node = ctx.insert_new_node_on_output("Transpose",
                                                       node.output[0], name=utils.make_name("where_op_added"))
        ctx.copy_shape(node.output[0], transpose_node.output[0])
        ctx.copy_dtype(node.output[0], transpose_node.output[0])
