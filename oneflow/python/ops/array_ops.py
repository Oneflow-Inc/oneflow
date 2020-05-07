from __future__ import absolute_import

from functools import reduce
import operator

import os
import oneflow as flow
import oneflow.python.framework.compile_context as compile_context
import oneflow.python.framework.remote_blob as remote_blob_util
import oneflow.python.framework.distribute as distribute_util
import oneflow.python.framework.id_util as id_util
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util

from oneflow.python.oneflow_export import oneflow_export


@oneflow_export("gather")
def gather(params, indices, validate_indices=None, axis=None, batch_dims=0, name=None):
    op_conf = op_conf_util.OperatorConf()
    if name is None:
        op_conf.name = id_util.UniqueStr("Gather_")
    else:
        op_conf.name = name

    if axis is None:
        axis = batch_dims

    if batch_dims > 0:
        if axis == batch_dims:
            setattr(op_conf.batch_gather_conf, "in", params.logical_blob_name)
            op_conf.batch_gather_conf.indices = indices.logical_blob_name
            op_conf.batch_gather_conf.out = "out"
        elif axis > batch_dims:
            raise NotImplementedError
        else:
            raise AttributeError
    elif params.has_batch_axis() == False and params.distribute is distribute_util.split(0):
        assert axis == 0
        assert batch_dims == 0
        setattr(op_conf.gather_ms0_conf, "in", params.logical_blob_name)
        op_conf.gather_ms0_conf.indices = indices.logical_blob_name
        op_conf.gather_ms0_conf.out = "out"
    else:
        setattr(op_conf.gather_conf, "in", params.logical_blob_name)
        op_conf.gather_conf.indices = indices.logical_blob_name
        op_conf.gather_conf.out = "out"
        op_conf.gather_conf.axis = axis

    compile_context.CurJobAddOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)


@oneflow_export("local_gather")
def local_gather(params, indices, axis=0, name=None):
    op_conf = op_conf_util.OperatorConf()
    setattr(op_conf, "name", name if name is not None else id_util.UniqueStr("LocalGather_"))
    if axis < 0:
        axis += len(params.shape)
    setattr(op_conf.local_gather_conf, "in", params.logical_blob_name)
    setattr(op_conf.local_gather_conf, "indices", indices.logical_blob_name)
    setattr(op_conf.local_gather_conf, "axis", axis)
    setattr(op_conf.local_gather_conf, "out", "out")
    compile_context.CurJobAddOp(op_conf)
    out_lbi = logical_blob_id_util.LogicalBlobId()
    setattr(out_lbi, "op_name", op_conf.name)
    setattr(out_lbi, "blob_name", "out")
    return remote_blob_util.RemoteBlob(out_lbi)

    def gather_lambda(params, indices):
        return gather(params, indices, axis=axis, name=name)

    return flow.advanced.distribute_map((params, indices), gather_lambda)

def infer_shape(x, shape):
    dim_index_need_infer = shape.index(-1) if shape.count(-1) == 1 else None
    in_elem_cnt = reduce(operator.mul, x.shape, 1)
    out_elem_cnt = reduce(operator.mul, shape, 1)
    if dim_index_need_infer is not None:
        assert (in_elem_cnt % out_elem_cnt) == 0
        shape[dim_index_need_infer] = int(abs(in_elem_cnt / out_elem_cnt))
    else:
        assert in_elem_cnt == out_elem_cnt
    return shape

@oneflow_export("reshape")
def reshape(x, shape, name=None):
    assert isinstance(shape, tuple) or isinstance(shape, list)
    shape = list(shape)
    assert all(dim == -1 or dim > 0 for dim in shape)
    assert shape.count(-1) <= 1
    if (not x.is_dynamic) and (os.getenv("ENABLE_USER_OP") == 'True'):
        if name is None:
            name = id_util.UniqueStr("Reshape_")
        return flow.user_op_builder(name).Op("reshape")\
            .Input("in", [x])\
            .Output("out")\
            .SetAttr("shape", infer_shape(x, shape), "AttrTypeShape")\
            .Build().RemoteBlobList()[0]
    else:
        op_conf = op_conf_util.OperatorConf()
        if x.is_dynamic:
            setattr(op_conf, "name", name if name is not None else id_util.UniqueStr("DynamicReshape_"))
            setattr(op_conf.dynamic_reshape_conf, "in", x.logical_blob_name)
            op_conf.dynamic_reshape_conf.shape.dim.extend(list(shape))
            setattr(op_conf.dynamic_reshape_conf, "out", "out")
        else:
            op_conf.name = id_util.UniqueStr("Reshape_" + x.op_name)
            setattr(op_conf.reshape_conf, "in", x.logical_blob_name)
            op_conf.reshape_conf.shape.dim[:] = list(infer_shape(x, shape))
            op_conf.reshape_conf.out = "out"
        compile_context.CurJobAddOp(op_conf)
        lbi = logical_blob_id_util.LogicalBlobId()
        lbi.op_name = op_conf.name
        lbi.blob_name = "out"
        return remote_blob_util.RemoteBlob(lbi)


@oneflow_export("reshape_like")
def reshape_like(x, like, name=None):
    op_conf = op_conf_util.OperatorConf()
    op_conf.name = id_util.UniqueStr("ReshapeLike_")
    setattr(op_conf.reshape_like_conf, "x", x.logical_blob_name)
    setattr(op_conf.reshape_like_conf, "like", like.logical_blob_name)
    op_conf.reshape_like_conf.y = "y"
    compile_context.CurJobAddOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "y"
    return remote_blob_util.RemoteBlob(lbi)


@oneflow_export("dynamic_reshape")
def dynamic_reshape(x, shape, name=None):
    assert isinstance(shape, tuple) or isinstance(shape, list)
    shape = list(shape)
    op_conf = op_conf_util.OperatorConf()
    setattr(op_conf, "name", name if name is not None else id_util.UniqueStr("DynamicReshape_"))
    setattr(op_conf.dynamic_reshape_conf, "in", x.logical_blob_name)
    op_conf.dynamic_reshape_conf.shape.dim.extend(list(shape))
    setattr(op_conf.dynamic_reshape_conf, "out", "out")
    compile_context.CurJobAddOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)


@oneflow_export("transpose")
def transpose(a, perm=None, conjugate=False, name=None):
    assert isinstance(perm, (tuple, list))

    if name is None:
        name = id_util.UniqueStr("Tranpose_")

    if conjugate:
        raise NotImplementedError

    if os.getenv("ENABLE_USER_OP") == 'True':
        return flow.user_op_builder(name).Op("transpose")\
            .Input("input", [a])\
            .Output("output")\
            .SetAttr("perm", perm, "AttrTypeListInt32")\
            .Build().RemoteBlobList()[0]
    else:
        op_conf = op_conf_util.OperatorConf()
        op_conf.name = name
        setattr(op_conf.transpose_conf, "in", a.logical_blob_name)
        op_conf.transpose_conf.out = "out"
        op_conf.transpose_conf.perm.extend(list(perm))

        compile_context.CurJobAddOp(op_conf)
        lbi = logical_blob_id_util.LogicalBlobId()
        lbi.op_name = op_conf.name
        lbi.blob_name = "out"
        return remote_blob_util.RemoteBlob(lbi)


@oneflow_export("slice")
def slice(input_, begin, size, name=None):
    r"""Extracts a slice from a tensor.

    Args:
        input_: A `Blob`.
        begin: A list or a tuple, indicate each dimension slice begin, whose length must be equal
            to input_'s number of dimensions, the first element of beign must be set to None.
            (because oneflow internal slice op do not support slice at dim0 at present)
        size: A list or a tuple, indicate each dimension slice size, whose length must be equal
            to input_'s number of dimensions, the first element of beign must be set to None.
        name: A name for the operation (optional).
    """
    ndims = len(input_.static_shape)
    assert (
        isinstance(begin, (list, tuple)) and len(begin) == ndims
    ), "begin must be a list or tuple whose length is the same with input_'s number of dimensions."
    assert (
        isinstance(size, (list, tuple)) and len(size) == ndims
    ), "size must be a list or tuple whose length is the same with input_'s number of dimensions."
    # assert (
    #     begin[0] is None
    # ), "begin not support dim0 slice at present, the first element of begin must be set to None"
    # assert (
    #     size[0] is None
    # ), "size not support dim0 slice at present, the first element of size must be set to None"

    slice_conf_list = []
    for b, s, d in list(zip(begin, size, input_.static_shape)):
        slice_conf = op_conf_util.DimSliceConf()
        if b is not None:
            if b < -d or b > d - 1:
                raise ValueError(
                    "'i'th element of begin must be greater than or equal to negative input_'s 'i'th dimension "
                    "and less than input_'s 'i'th dimension."
                )
            b = b + d if b < 0 else b
            slice_conf.start = b
        if s is not None:
            if s > 0:
                if b + s > d:
                    raise ValueError(
                        "the sum of 'i'th element of begin and 'i'th element of size must be "
                        "less than or equal to input_'s 'i'th dimension."
                    )
                slice_conf.end = b + s
            elif s == -1:
                slice_conf.end = d
            else:
                raise ValueError(
                    "elements of size must be an int that greater then 0 or equal to -1"
                )
            slice_conf.stride = 1
        slice_conf_list.append(slice_conf)

    op_conf = op_conf_util.OperatorConf()
    setattr(op_conf, "name", name if name is not None else id_util.UniqueStr("Slice_"))
    setattr(op_conf.slice_conf, "in", input_.logical_blob_name)
    setattr(op_conf.slice_conf, "out", "out")
    op_conf.slice_conf.dim_slice_conf.extend(slice_conf_list)

    compile_context.CurJobAddOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)


@oneflow_export("slice_v2")
def slice_v2(input, slice_tup_list, name=None):
    r"""Extracts a slice from a tensor.
    Args:
        input: A `Blob`.
        slice_tup_list: A list of tuple, indicate each dimension slice (begin, end, stride). 
            Note: The function don't support slice at dim0 for now , first element of slice_tup_list must be 
            (None, None, None).
        name: A name for the operation (optional).
    """
    name = name or id_util.UniqueStr("SliceV2_")
    if not isinstance(name, str):
        raise ValueError('param "name" must be a string')

    ndims = len(input.shape)
    if not isinstance(slice_tup_list, (list, tuple)) or len(slice_tup_list) > ndims:
        raise ValueError(
            'param "slice_tup_list" must be a list or tuple whose length should be '
            "less than or equal to number of dimensions of input"
        )

    # if length of slice_tup_list is less than number of dimensions of input, fill it to length of ndims reduce 1
    if len(slice_tup_list) < ndims:
        slice_tup_list += [(None, None, None)] * (ndims - len(slice_tup_list))

    begin_list = []
    end_list = []
    stride_list = []
    has_begin_list = []
    has_end_list = []
    for slice_tup, dim in zip(slice_tup_list, input.shape):
        if not isinstance(slice_tup, (tuple, list)):
            raise ValueError(
                "element of slice_tup_list must be a list or tuple with form (begin, end, stride)"
            )
        (begin, end, stride) = slice_tup
        has_begin = 1
        has_end = 1
        if begin is None:
            begin = 0
            has_begin = 0
        if end is None:
            end = 0
            has_end = 0
        if stride is None:
            stride = 1
        begin_list.append(begin)
        end_list.append(end)
        stride_list.append(stride)
        has_begin_list.append(has_begin)
        has_end_list.append(has_end)

    op = (
        flow.user_op_builder(name)
        .Op("slice_v2")
        .Input("x", [input])
        .Output("y")
        .SetAttr("begin", begin_list, "AttrTypeListInt64")
        .SetAttr("end", end_list, "AttrTypeListInt64")
        .SetAttr("stride", stride_list, "AttrTypeListInt64")
        .SetAttr("has_begin", has_begin_list, "AttrTypeListInt64")
        .SetAttr("has_end", has_end_list, "AttrTypeListInt64")
        .Build()
    )
    return op.RemoteBlobList()[0]


@oneflow_export("concat")
def concat(values, axis, name=None):
    op_conf = op_conf_util.OperatorConf()
    setattr(op_conf, "name", name if name is not None else id_util.UniqueStr("Concat_"))
    op_conf.concat_conf.out = "out"
    if not isinstance(values, (list, tuple)):
        values = [values]
    getattr(op_conf.concat_conf, "in").extend([v.logical_blob_name for v in values])
    op_conf.concat_conf.axis = axis
    compile_context.CurJobAddOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)


@oneflow_export("gather_nd")
def gather_nd(params, indices, name=None):
    if name is None:
        name = id_util.UniqueStr("GatherNd_")
    op = (
        flow.user_op_builder(name)
        .Op("gather_nd")
        .Input("params", [params])
        .Input("indices", [indices])
        .Output("out")
        .Build()
    )
    return op.RemoteBlobList()[0]


@oneflow_export("scatter_nd")
def scatter_nd(indices, updates, shape, name=None):
    if name is None:
        name = id_util.UniqueStr("ScatterNd_")
    op = (
        flow.user_op_builder(name)
        .Op("scatter_nd")
        .Input("indices", [indices])
        .Input("updates", [updates])
        .SetAttr("shape", shape, "AttrTypeShape")
        .Output("out")
        .Build()
    )
    return op.RemoteBlobList()[0]


@oneflow_export("tensor_scatter_nd_update")
def tensor_scatter_nd_update(params, indices, updates, name=None):
    if name is None:
        name = id_util.UniqueStr("TensorScatterNdUpdate_")
    op = (
        flow.user_op_builder(name)
        .Op("tensor_scatter_nd_update")
        .Input("params", [params])
        .Input("updates", [updates])
        .Input("indices", [indices])
        .Output("out")
        .Build()
    )
    return op.RemoteBlobList()[0]


@oneflow_export("tensor_scatter_nd_add")
def tensor_scatter_nd_add(params, indices, updates, name=None):
    if name is None:
        name = id_util.UniqueStr("TensorScatterNdAdd_")
    op = (
        flow.user_op_builder(name)
        .Op("tensor_scatter_nd_add")
        .Input("params", [params])
        .Input("updates", [updates])
        .Input("indices", [indices])
        .Output("out")
        .Build()
    )
    return op.RemoteBlobList()[0]


@oneflow_export("argwhere")
def argwhere(condition, dtype=None, name=None):
    if name is None:
        name = id_util.UniqueStr("ArgWhere_")

    op_conf = op_conf_util.OperatorConf()
    setattr(op_conf, "name", name)
    setattr(op_conf.arg_where_conf, "in", condition.logical_blob_name)
    setattr(op_conf.arg_where_conf, "out", "out")
    setattr(op_conf.arg_where_conf, "out_size", "out_size")
    if dtype is not None:
        setattr(op_conf.arg_where_conf, "data_type", dtype)
    compile_context.CurJobAddOp(op_conf)

    arg_where_out_lbi = logical_blob_id_util.LogicalBlobId()
    setattr(arg_where_out_lbi, "op_name", op_conf.name)
    setattr(arg_where_out_lbi, "blob_name", "out")

    arg_where_out_size_lbi = logical_blob_id_util.LogicalBlobId()
    setattr(arg_where_out_size_lbi, "op_name", op_conf.name)
    setattr(arg_where_out_size_lbi, "blob_name", "out_size")

    arg_where_out = remote_blob_util.RemoteBlob(arg_where_out_lbi)
    arg_where_out_size = remote_blob_util.RemoteBlob(arg_where_out_size_lbi)
    return sync_dynamic_resize(arg_where_out, arg_where_out_size)


@oneflow_export("nonzero")
def nonzero(a, name=None):
    if name is None:
        argwhere_name = id_util.UniqueStr("Nonzero_ArgWhere_")
        tranpose_name = id_util.UniqueStr("Nonzero_Transpose_")
    else:
        argwhere_name = name + "_ArgWhere"
        tranpose_name = name + "_Transpose"
    indices = argwhere(a, name=argwhere_name)
    return transpose(indices, perm=(1, 0), name=tranpose_name)


@oneflow_export("where")
def where(condition, x=None, y=None, name=None): 
    if x is None and y is None:
        return argwhere(condition, name=name)
    elif x is not None and y is not None:
        if name is None:
            name = id_util.UniqueStr("Where_")

        broadcast_cond = flow.broadcast_to_compatible_with(condition, [x, y])
        broadcast_x = flow.broadcast_to_compatible_with(x, [condition, y])
        broadcast_y = flow.broadcast_to_compatible_with(y, [condition, x])
        return (
            flow.user_op_builder(name)
            .Op("where")
            .Input("condition", [broadcast_cond])
            .Input("x", [broadcast_x])
            .Input("y", [broadcast_y])
            .Output("out")
            .Build()
            .RemoteBlobList()[0]
        )
    else:
        raise ValueError("it is not supported when exactly one of x or y is non-None")


@oneflow_export("piece_slice")
def piece_slice(inputs, output_size, name=None):
    assert inputs.shape[0] == output_size
    op_conf = op_conf_util.OperatorConf()
    setattr(op_conf, "name", name if name is not None else id_util.UniqueStr("PieceSlice_"))
    setattr(op_conf.piece_slice_conf, "in", inputs.logical_blob_name)
    op_conf.piece_slice_conf.out.extend(["out_" + str(i) for i in range(output_size)])
    compile_context.CurJobAddOp(op_conf)
    ret = []
    for i in range(output_size):
        out_lbi = logical_blob_id_util.LogicalBlobId()
        setattr(out_lbi, "op_name", op_conf.name)
        setattr(out_lbi, "blob_name", "out_" + str(i))
        ret.append(remote_blob_util.RemoteBlob(out_lbi))
    return tuple(ret)


@oneflow_export("elem_cnt")
def elem_cnt(inputs, dtype=None, name=None):
    op_conf = op_conf_util.OperatorConf()
    setattr(op_conf, "name", name if name is not None else id_util.UniqueStr("ElemCnt_"))
    op_conf.shape_elem_cnt_conf.x = inputs.logical_blob_name

    op_conf.shape_elem_cnt_conf.exclude_axis_conf.SetInParent()
    if dtype is not None:
        op_conf.shape_elem_cnt_conf.data_type = dtype
    op_conf.shape_elem_cnt_conf.y = "y"
    compile_context.CurJobAddOp(op_conf)
    out_lbi = logical_blob_id_util.LogicalBlobId()
    setattr(out_lbi, "op_name", op_conf.name)
    setattr(out_lbi, "blob_name", "y")
    return remote_blob_util.RemoteBlob(out_lbi)


@oneflow_export("sync_dynamic_resize")
def sync_dynamic_resize(inputs, size, name=None):
    op_conf = op_conf_util.OperatorConf()
    setattr(op_conf, "name", name if name is not None else id_util.UniqueStr("SyncDynamicResize_"))
    setattr(op_conf.sync_dynamic_resize_conf, "in", inputs.logical_blob_name)
    setattr(op_conf.sync_dynamic_resize_conf, "size", size.logical_blob_name)
    setattr(op_conf.sync_dynamic_resize_conf, "axis", 0)
    setattr(op_conf.sync_dynamic_resize_conf, "out", "out")
    compile_context.CurJobAddOp(op_conf)
    out_lbi = logical_blob_id_util.LogicalBlobId()
    setattr(out_lbi, "op_name", op_conf.name)
    setattr(out_lbi, "blob_name", "out")
    return remote_blob_util.RemoteBlob(out_lbi)


@oneflow_export("stack")
def stack(inputs, axis, name=None):
    if not isinstance(inputs, (list, tuple)):
        inputs = [inputs]

    if axis < 0:
        axis = axis + len(inputs[0].shape)

    assert axis == 0, "Only support dim0 stack now."

    op_conf = op_conf_util.OperatorConf()
    setattr(op_conf, "name", name or id_util.UniqueStr("Stack_"))
    getattr(op_conf.stack_conf, "in").extend([input.logical_blob_name for input in inputs])
    setattr(op_conf.stack_conf, "axis", axis)
    setattr(op_conf.stack_conf, "out", "out")
    compile_context.CurJobAddOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)


@oneflow_export("assign")
def assign(ref, value, dtype=None, name=None):
    op_conf = op_conf_util.OperatorConf()
    setattr(op_conf, "name", name if name is not None else id_util.UniqueStr("Assign_"))
    op_conf.assign_conf.ref = ref.logical_blob_name
    op_conf.assign_conf.value = value.logical_blob_name
    compile_context.CurJobAddOp(op_conf)
    out_lbi = logical_blob_id_util.LogicalBlobId()
    setattr(out_lbi, "op_name", op_conf.name)
    setattr(out_lbi, "blob_name", "y")
    return remote_blob_util.RemoteBlob(out_lbi)

@oneflow_export("random.generate_random_batch_permutation_indices")
def generate_random_batch_permutation_indices(value, seed=None, name=None):
    import random
    op = (
        flow.user_op_builder(name if name is not None else id_util.UniqueStr(value.op_name + "_random_batch_permutation_indices"))
        .Op("generate_random_batch_permutation_indices")
        .Input("x", [value])
        .Output("y")
    )
    if seed is not None:
        op.SetAttr("seed", seed, "AttrTypeInt64")
    else:
        op.SetAttr("seed", random.randint(-2**63 + 1, 2**63 - 1), "AttrTypeInt64")
    return (
        op
        .Build()
        .RemoteBlobList()[0]
    )

@oneflow_export("random.shuffle")
def shuffle(value, seed=None, name=None):
    return flow.gather(
        value, generate_random_batch_permutation_indices(value, seed)
    )


@oneflow_export("identity")
def identity(x, name=None):
    if name is None:
        name = id_util.UniqueStr("Identity_")
    op_conf = op_conf_util.OperatorConf()
    op_conf.name = name
    setattr(op_conf.identity_conf, "in", x.logical_blob_name)
    op_conf.identity_conf.out = "out"
    compile_context.CurJobAddOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)


@oneflow_export("squeeze")
def squeeze(input, axis=None, name=None):
    if axis is None:
        axis = [idx for idx, dim in enumerate(input.shape) if dim is 1]
    else:
        assert isinstance(axis, list) or isinstance(axis, tuple)
        in_num_axes = len(input.shape)
        for x in axis:
            assert x >= -in_num_axes and x < in_num_axes
    return (
        flow.user_op_builder(name if name is not None else id_util.UniqueStr("Squeeze_"))
        .Op("squeeze")
        .Input("in", [input])
        .Output("out")
        .SetAttr("axes", list(axis), "AttrTypeListInt32")
        .Build()
        .RemoteBlobList()[0]
    )


@oneflow_export("expand_dims")
def expand_dims(input, axis, name=None):
    in_num_axes = len(input.shape)
    assert axis >= -(in_num_axes + 1) and axis <= in_num_axes
    return (
        flow.user_op_builder(name if name is not None else id_util.UniqueStr("ExpandDims_"))
        .Op("expand_dims")
        .Input("in", [input])
        .Output("out")
        .SetAttr("axis", axis, "AttrTypeInt32")
        .Build()
        .RemoteBlobList()[0]
    )
