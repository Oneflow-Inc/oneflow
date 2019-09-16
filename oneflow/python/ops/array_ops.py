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
        assert (reduce(operator.mul, x.shape, 1) %
                reduce(operator.mul, shape, 1)) == 0
        shape[dim_index_need_infer] = int(
            abs(reduce(operator.mul, x.shape, 1) / reduce(operator.mul, shape, 1)))
    else:
        assert reduce(operator.mul, x.shape, 1) == reduce(
            operator.mul, shape, 1)
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
        op_conf, "name", name if name is not None else id_util.UniqueStr("Slice_"),
    )
    setattr(op_conf.slice_conf, "in", input_.logical_blob_name)
    setattr(op_conf.slice_conf, "out", "out")
    op_conf.slice_conf.dim_slice_conf.extend(slice_conf_list)

    compile_context.CurJobAddOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)


@oneflow_export("constant")
def constant(
    value,
    dtype=None,
    shape=None,
    name=None,
):
    op_conf = op_conf_util.OperatorConf()
    setattr(
        op_conf,
        "name",
        name if name is not None else id_util.UniqueStr("Constant_"),
    )
    assert value is not None
    assert dtype is not None
    if isinstance(value, list):
        raise NotImplementedError
    elif isinstance(value, (int, float)):
        # TODO: should only set dtype once 
        setattr(op_conf.constant_conf, "data_type", dtype)
        op_conf.constant_conf.initializer.CopyFrom(
            flow.constant_initializer(value, dtype)
        )
    else:
        raise NotImplementedError

    if shape is not None:
        assert isinstance(shape, (list, tuple))
        op_conf.constant_conf.shape.dim.extend(list(shape))

    setattr(op_conf.constant_conf, "out", "out")
    compile_context.CurJobAddOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)


@oneflow_export('concat')
def concat(values,
           axis,
           name=None):
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
        [v.logical_blob_name for v in values])
    op_conf.concat_conf.axis = axis
    compile_context.CurJobAddOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)
