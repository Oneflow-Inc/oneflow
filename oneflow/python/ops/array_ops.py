from __future__ import absolute_import

from functools import reduce
import operator

import oneflow as flow
import oneflow.python.framework.compile_context as compile_context
import oneflow.python.framework.remote_blob as remote_blob_util
import oneflow.python.framework.distribute as distribute_util
import oneflow.python.framework.id_util as id_util
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util

from oneflow.python.oneflow_export import oneflow_export


@oneflow_export("gather")
def gather(
    params, indices, validate_indices=None, axis=None, batch_dims=0, name=None
):
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
    elif params.distribute is distribute_util.split(0):
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


@oneflow_export("reshape")
def reshape(x, shape, name=None):
    assert isinstance(shape, tuple) or isinstance(shape, list)
    shape = list(shape)
    assert all(dim == -1 or dim > 0 for dim in shape)
    assert shape.count(-1) <= 1
    dim_index_need_infer = shape.index(-1) if shape.count(-1) == 1 else None
    if dim_index_need_infer is not None:
        assert (
            reduce(operator.mul, x.shape, 1) % reduce(operator.mul, shape, 1)
        ) == 0
        shape[dim_index_need_infer] = int(
            abs(
                reduce(operator.mul, x.shape, 1)
                / reduce(operator.mul, shape, 1)
            )
        )
    else:
        assert reduce(operator.mul, x.shape, 1) == reduce(
            operator.mul, shape, 1
        )
    op_conf = op_conf_util.OperatorConf()
    op_conf.name = id_util.UniqueStr("Reshape_")
    setattr(op_conf.reshape_conf, "in", x.logical_blob_name)
    op_conf.reshape_conf.shape.dim[:] = list(shape)
    op_conf.reshape_conf.out = "out"
    compile_context.CurJobAddOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)


@oneflow_export("dynamic_reshape")
def dynamic_reshape(x, shape, name=None):
    return reshape(x, shape)
    assert isinstance(shape, tuple) or isinstance(shape, list)
    shape = list(shape)
    op_conf = op_conf_util.OperatorConf()
    setattr(
        op_conf,
        "name",
        name if name is not None else id_util.UniqueStr("DynamicReshape_"),
    )
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
    assert (
        begin[0] is None
    ), "begin not support dim0 slice at present, the first element of begin must be set to None"
    assert (
        size[0] is None
    ), "size not support dim0 slice at present, the first element of size must be set to None"

    slice_conf_list = []
    # ignore first dimension because it's not supported yet
    for b, s, d in list(zip(begin, size, input_.static_shape))[1:]:
        slice_conf = op_conf_util.DimSliceConf()
        if b < -d or b > d - 1:
            raise ValueError(
                "'i'th element of begin must be greater than or equal to negative input_'s 'i'th dimension "
                "and less than input_'s 'i'th dimension."
            )
        b = b + d if b < 0 else b
        slice_conf.start = b

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
    setattr(
        op_conf,
        "name",
        name if name is not None else id_util.UniqueStr("Slice_"),
    )
    setattr(op_conf.slice_conf, "in", input_.logical_blob_name)
    setattr(op_conf.slice_conf, "out", "out")
    op_conf.slice_conf.dim_slice_conf.extend(slice_conf_list)

    compile_context.CurJobAddOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)


@oneflow_export("concat")
def concat(values, axis, name=None):
    op_conf = op_conf_util.OperatorConf()
    setattr(
        op_conf,
        "name",
        name if name is not None else id_util.UniqueStr("Concat_"),
    )
    op_conf.concat_conf.out = "out"
    if not isinstance(values, (list, tuple)):
        values = [values]
    getattr(op_conf.concat_conf, "in").extend(
        [v.logical_blob_name for v in values]
    )
    op_conf.concat_conf.axis = axis
    compile_context.CurJobAddOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)


@oneflow_export("local_scatter_nd_update")
def local_scatter_nd_update(inputs, indices, updates, name=None):
    op_conf = op_conf_util.OperatorConf()
    setattr(
        op_conf,
        "name",
        name
        if name is not None
        else id_util.UniqueStr("LocalScatterNdUpdate_"),
    )
    setattr(
        op_conf.local_scatter_nd_update_conf, "in", inputs.logical_blob_name
    )
    setattr(
        op_conf.local_scatter_nd_update_conf,
        "indices",
        indices.logical_blob_name,
    )
    setattr(
        op_conf.local_scatter_nd_update_conf,
        "updates",
        updates.logical_blob_name,
    )
    setattr(op_conf.local_scatter_nd_update_conf, "out", "out")
    compile_context.CurJobAddOp(op_conf)
    out_lbi = logical_blob_id_util.LogicalBlobId()
    setattr(out_lbi, "op_name", op_conf.name)
    setattr(out_lbi, "blob_name", "out")
    return remote_blob_util.RemoteBlob(out_lbi)


@oneflow_export("local_gather")
def local_gather(params, indices, axis=0, name=None):
    return gather(params, indices, axis, name)
    op_conf = op_conf_util.OperatorConf()
    setattr(
        op_conf,
        "name",
        name if name is not None else id_util.UniqueStr("LocalGather_"),
    )
    setattr(op_conf.local_gather_conf, "in", params.logical_blob_name)
    setattr(op_conf.local_gather_conf, "indices", indices.logical_blob_name)
    setattr(op_conf.local_gather_conf, "axis", axis)
    setattr(op_conf.local_gather_conf, "out", "out")
    compile_context.CurJobAddOp(op_conf)
    out_lbi = logical_blob_id_util.LogicalBlobId()
    setattr(out_lbi, "op_name", op_conf.name)
    setattr(out_lbi, "blob_name", "out")
    return remote_blob_util.RemoteBlob(out_lbi)


@oneflow_export("local_nonzero")
def local_nonzero(input, name=None):
    op_conf = op_conf_util.OperatorConf()
    setattr(
        op_conf,
        "name",
        name if name is not None else id_util.UniqueStr("LocalNonzero_"),
    )
    setattr(op_conf.local_nonzero_conf, "in", input.logical_blob_name)
    setattr(op_conf.local_nonzero_conf, "out", "out")
    compile_context.CurJobAddOp(op_conf)
    out_lbi = logical_blob_id_util.LogicalBlobId()
    setattr(out_lbi, "op_name", op_conf.name)
    setattr(out_lbi, "blob_name", "out")
    return remote_blob_util.RemoteBlob(out_lbi)


@oneflow_export("where")
def where(condition, x, y, name=None):
    assert condition.shape == x.shape and x.shape == y.shape
    op_conf = op_conf_util.OperatorConf()
    setattr(
        op_conf,
        "name",
        name if name is not None else id_util.UniqueStr("Where_"),
    )
    setattr(op_conf.where_conf, "condition", condition.logical_blob_name)
    setattr(op_conf.where_conf, "lhs", x.logical_blob_name)
    setattr(op_conf.where_conf, "rhs", y.logical_blob_name)
    setattr(op_conf.where_conf, "out", "out")
    compile_context.CurJobAddOp(op_conf)
    out_lbi = logical_blob_id_util.LogicalBlobId()
    setattr(out_lbi, "op_name", op_conf.name)
    setattr(out_lbi, "blob_name", "out")
    return remote_blob_util.RemoteBlob(out_lbi)


@oneflow_export("squeeze")
def squeeze(inputs, axis, name=None):
    assert isinstance(axis, list)
    op_conf = op_conf_util.OperatorConf()
    setattr(
        op_conf,
        "name",
        name if name is not None else id_util.UniqueStr("Squeeze_"),
    )
    assert all(axis_i == -1 or axis_i > 0 for axis_i in axis)
    shape = []
    for i, dim in enumerate(inputs.shape):
        if i in axis:
            assert dim == 1
        else:
            shape.append(dim)
    return reshape(inputs, shape)


@oneflow_export("expand_dims")
def expand_dims(inputs, axis, name=None):
    new_shape = list(inputs.shape)
    new_shape.insert(axis, 1)
    return reshape(inputs, new_shape)


@oneflow_export("piece_slice")
def piece_slice(inputs, output_size, name=None):
    expanded_inputs = reshape(inputs, [1] + list(inputs.shape))
    size = [None, 1] + list(inputs.shape)[1:]
    ret = []
    for i in range(output_size):
        begin = [None, i] + [0] * (len(inputs.shape) - 1)
        output = slice(expanded_inputs, begin, size)
        squeezed_output = reshape(output, list(output.shape)[2:])
        ret.append(squeezed_output)
    return ret
    op_conf = op_conf_util.OperatorConf()
    setattr(
        op_conf,
        "name",
        name if name is not None else id_util.UniqueStr("PieceSlice_"),
    )
    setattr(op_conf.piece_slice_conf, "in", inputs.logical_blob_name)
    op_conf.piece_slice_conf.out.extend(
        ["out_" + str(i) for i in range(output_size)]
    )
    compile_context.CurJobAddOp(op_conf)
    ret = []
    for i in range(output_size):
        out_lbi = logical_blob_id_util.LogicalBlobId()
        setattr(out_lbi, "op_name", op_conf.name)
        setattr(out_lbi, "blob_name", "out_" + str(i))
        ret.append(remote_blob_util.RemoteBlob(out_lbi))
    return tuple(ret)


@oneflow_export("elem_cnt")
def elem_cnt(inputs, begin_axis=None, end_axis=None, data_type=None, name=None):
    op_conf = op_conf_util.OperatorConf()
    setattr(
        op_conf,
        "name",
        name if name is not None else id_util.UniqueStr("ElemCnt_"),
    )
    setattr(op_conf.element_count_conf, "in", inputs.logical_blob_name)
    if begin_axis is not None:
        op_conf.element_count_conf.begin_axis = begin_axis
    if end_axis is not None:
        op_conf.element_count_conf.end_axis = end_axis
    if data_type is not None:
        op_conf.element_count_conf.data_type = data_type
    op_conf.element_count_conf.out = "out"
    compile_context.CurJobAddOp(op_conf)
    out_lbi = logical_blob_id_util.LogicalBlobId()
    setattr(out_lbi, "op_name", op_conf.name)
    setattr(out_lbi, "blob_name", "out")
    return remote_blob_util.RemoteBlob(out_lbi)
