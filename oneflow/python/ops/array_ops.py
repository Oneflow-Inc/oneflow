from __future__ import absolute_import

import oneflow as flow
import oneflow.python.framework.compile_context as compile_context
import oneflow.python.framework.remote_blob as remote_blob_util
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
    ndims = len(input_.static_shape)
    assert (
        isinstance(begin, (list, tuple)) and len(begin) == ndims
    ), "begin must be a list or tuple whose length is the same with input_'s number of dimensions."
    assert (
        isinstance(size, (list, tuple)) and len(size) == ndims
    ), "size must be a list or tuple whose length is the same with input_'s number of dimensions."

    if name is None:
        name = id_util.UniqueStr("Slice_")

    slice_conf_list = []
    for b, s, d in zip(begin, size, input_.static_shape):
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
    op_conf.name = name
    setattr(op_conf.slice_conf, "in", input_.logical_blob_name)
    op_conf.slice_conf.out = "out"
    # ignore first slice conf because oneflow slice op not support dim0 slice yet
    op_conf.slice_conf.dim_slice_conf.extend(slice_conf_list[1:])

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
    # verify_shape=False
):
    op_conf = op_conf_util.OperatorConf()

    if name is None:
        op_conf.name = id_util.UniqueStr("Constant_")
    else:
        op_conf.name = name

    if value is not None:
        if isinstance(value, list):
            raise NotImplementedError
        elif isinstance(value, (int, float)):
            op_conf.constant_conf.initializer.CopyFrom(
                flow.constant_initializer(value, dtype)
            )
        else:
            raise NotImplementedError

    if dtype is not None:
        setattr(op_conf.constant_conf, "data_type", dtype)

    if shape is not None:
        assert isinstance(shape, (list, tuple))
        op_conf.constant_conf.shape.dim.extend(list(shape))

    setattr(op_conf.constant_conf, "out", "out")
    compile_context.CurJobAddOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)
