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


@oneflow_export("reshape")
def reshape(x, shape, name=None):
    assert isinstance(shape, tuple) or isinstance(shape, list)
    shape = list(shape)
    assert all(dim == -1 or dim > 0 for dim in shape)
    assert shape.count(-1) <= 1
    dim_index_need_infer = shape.index(-1) if shape.count(-1) == 1 else None
    if dim_index_need_infer is not None:
        assert (reduce(operator.mul, x.shape, 1) % reduce(operator.mul, shape, 1)) == 0
        shape[dim_index_need_infer] = int(
            abs(reduce(operator.mul, x.shape, 1) / reduce(operator.mul, shape, 1))
        )
    else:
        assert reduce(operator.mul, x.shape, 1) == reduce(operator.mul, shape, 1)
    op_conf = op_conf_util.OperatorConf()
    op_conf.name = id_util.UniqueStr("Reshape_" + x.op_name)
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
    ndims = len(input_.shape)
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

    slice_tup_list = []
    for begin_, size_, dims in list(zip(begin, size, input_.shape)):
        if begin_ is not None:
            if begin_ < -dims or begin_ > dims - 1:
                raise ValueError(
                    "'i'th element of begin must be greater than or equal to negative input_'s 'i'th dimension "
                    "and less than input_'s 'i'th dimension."
                )
            begin_ = begin_ if begin_ >= 0 else begin_ + dims

        end_ = None
        if size_ is not None:
            if size_ > 0 and begin_ + size_ > dims:
                raise ValueError(
                    "the sum of 'i'th element of begin and 'i'th element of size must be "
                    "less than or equal to input_'s 'i'th dimension."
                )
            if size_ <= 0:
                end_ = ndims
            else:
                end_ = begin_ + size_

        slice_tup_list.append((begin_, end_, 1))

    return slice_v3(input_, slice_tup_list, name)


@oneflow_export("slice_v2")
# slice_confs: list of tuple/list (begin, end, stride)
def slice_v2(input, slice_confs, name=None):
    op_conf = op_conf_util.OperatorConf()
    setattr(op_conf, "name", name if name is not None else id_util.UniqueStr("SliceV2_"))
    setattr(op_conf.slice_v2_conf, "in", input.logical_blob_name)
    setattr(op_conf.slice_v2_conf, "out", "out")
    slice_conf_list = []
    for dim_slice_conf in slice_confs:
        assert isinstance(dim_slice_conf, dict)
        slice_conf = op_conf_util.DimSliceConf()
        if "begin" in dim_slice_conf:
            slice_conf.start = dim_slice_conf["begin"]
        if "end" in dim_slice_conf:
            slice_conf.end = dim_slice_conf["end"]
        if "stride" in dim_slice_conf:
            slice_conf.stride = dim_slice_conf["stride"]
        else:
            slice_conf.stride = 1
        slice_conf_list.append(slice_conf)
    op_conf.slice_v2_conf.dim_slice_conf.extend(slice_conf_list)
    compile_context.CurJobAddOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)


@oneflow_export("slice_v3")
def slice_v3(input, slice_tup_list, name=None):
    r"""Extracts a slice from a tensor.

    Args:
        input: A `Blob`.
        slice_tup_list: A list of tuple, indicate each dimension slice (begin, end, stride). 
            (The function don't support slice at dim0 now , first element of slice_tup_list must be 
            (None, None, None). )
        name: A name for the operation (optional).
    """
    ndims = len(input.shape)
    if not isinstance(slice_tup_list, (list, tuple)) or len(slice_tup_list) > ndims:
        raise ValueError

    if len(slice_tup_list) < ndims:
        slice_tup_list += [(None, None, None)] * (ndims - len(slice_tup_list))

    slice_conf_list = []
    for slice_tup, dims in list(zip(slice_tup_list, input.shape))[1:]:
        if not isinstance(slice_tup, (tuple, list)) or len(slice_tup) != 3:
            raise ValueError("slice_tup must be a (begin, end, stride) composed tuple or list")

        (begin, end, stride) = slice_tup
        slice_conf = op_conf_util.DimSliceConf()
        if begin is None:
            begin = 0
        if begin < -dims or begin > dims - 1:
            raise ValueError(
                "begin must be greater than or equal to negative dimensions "
                "and less than dimensions."
            )
        begin = begin if begin >= 0 else begin + dims

        if end is None:
            end = dims
        if end < -dims or end > dims:
            raise ValueError(
                "the sum of 'i'th element of begin and 'i'th element of size must be "
                "less than or equal to input_'s 'i'th dimension."
            )
        end = end if end >= 0 else end + dims

        if stride is None:
            stride = 1
        if stride == 0:
            raise ValueError("stride cannot be 0")

        slice_conf.start = begin
        slice_conf.end = end
        slice_conf.stride = stride
        slice_conf_list.append(slice_conf)

    op_conf = op_conf_util.OperatorConf()
    setattr(op_conf, "name", name if name is not None else id_util.UniqueStr("Slice_"))
    setattr(op_conf.slice_v3_conf, "in", input.logical_blob_name)
    setattr(op_conf.slice_v3_conf, "out", "out")
    op_conf.slice_v3_conf.dim_slice_conf.extend(slice_conf_list)

    compile_context.CurJobAddOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)


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


@oneflow_export("local_scatter_nd_update")
def local_scatter_nd_update(inputs, indices, updates, name=None):
    op_conf = op_conf_util.OperatorConf()
    setattr(
        op_conf, "name", name if name is not None else id_util.UniqueStr("LocalScatterNdUpdate_")
    )
    setattr(op_conf.local_scatter_nd_update_conf, "in", inputs.logical_blob_name)
    setattr(op_conf.local_scatter_nd_update_conf, "indices", indices.logical_blob_name)
    setattr(op_conf.local_scatter_nd_update_conf, "updates", updates.logical_blob_name)
    setattr(op_conf.local_scatter_nd_update_conf, "out", "out")
    compile_context.CurJobAddOp(op_conf)
    out_lbi = logical_blob_id_util.LogicalBlobId()
    setattr(out_lbi, "op_name", op_conf.name)
    setattr(out_lbi, "blob_name", "out")
    return remote_blob_util.RemoteBlob(out_lbi)


@oneflow_export("local_nonzero")
def local_nonzero(input, name=None):
    op_conf = op_conf_util.OperatorConf()
    setattr(op_conf, "name", name if name is not None else id_util.UniqueStr("LocalNonzero_"))
    setattr(op_conf.local_nonzero_conf, "in", input.logical_blob_name)
    setattr(op_conf.local_nonzero_conf, "out", "out")
    setattr(op_conf.local_nonzero_conf, "num_nonzero", "num_nonzero")
    compile_context.CurJobAddOp(op_conf)
    ret = []
    for obn in ["out", "num_nonzero"]:
        out_lbi = logical_blob_id_util.LogicalBlobId()
        setattr(out_lbi, "op_name", op_conf.name)
        setattr(out_lbi, "blob_name", obn)
        ret.append(remote_blob_util.RemoteBlob(out_lbi))
    return sync_dynamic_resize(ret[0], ret[1])


@oneflow_export("where")
def where(condition, x, y, name=None):
    assert condition.shape == x.shape and x.shape == y.shape
    op_conf = op_conf_util.OperatorConf()
    setattr(op_conf, "name", name if name is not None else id_util.UniqueStr("Where_"))
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
    setattr(op_conf, "name", name if name is not None else id_util.UniqueStr("Squeeze_"))
    assert all(axis_i == -1 or axis_i >= 0 for axis_i in axis)
    setattr(op_conf.squeeze_conf, "in", inputs.logical_blob_name)
    op_conf.squeeze_conf.out = "out"
    op_conf.squeeze_conf.axis.extend(axis)
    compile_context.CurJobAddOp(op_conf)
    out_lbi = logical_blob_id_util.LogicalBlobId()
    setattr(out_lbi, "op_name", op_conf.name)
    setattr(out_lbi, "blob_name", "out")
    return remote_blob_util.RemoteBlob(out_lbi)


@oneflow_export("expand_dims")
def expand_dims(inputs, axis, name=None):
    assert isinstance(axis, int)
    if axis < 0:
        axis = len(inputs.shape) + axis
    assert axis <= len(inputs.shape)
    op_conf = op_conf_util.OperatorConf()
    setattr(op_conf, "name", name if name is not None else id_util.UniqueStr("Expandims_"))
    setattr(op_conf.expand_dims_conf, "in", inputs.logical_blob_name)
    op_conf.expand_dims_conf.out = "out"
    op_conf.expand_dims_conf.axis = axis
    compile_context.CurJobAddOp(op_conf)
    out_lbi = logical_blob_id_util.LogicalBlobId()
    setattr(out_lbi, "op_name", op_conf.name)
    setattr(out_lbi, "blob_name", "out")
    return remote_blob_util.RemoteBlob(out_lbi)


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
def elem_cnt(inputs, begin_axis=None, end_axis=None, dtype=None, name=None):
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
def assign(ref, value, begin_axis=None, end_axis=None, dtype=None, name=None):
    op_conf = op_conf_util.OperatorConf()
    setattr(op_conf, "name", name if name is not None else id_util.UniqueStr("Assign_"))
    op_conf.assign_conf.ref = ref.logical_blob_name
    op_conf.assign_conf.value = value.logical_blob_name
    compile_context.CurJobAddOp(op_conf)
    out_lbi = logical_blob_id_util.LogicalBlobId()
    setattr(out_lbi, "op_name", op_conf.name)
    setattr(out_lbi, "blob_name", "y")
    return remote_blob_util.RemoteBlob(out_lbi)


@oneflow_export("random_like")
def random_like(like, seed=None, name=None):
    op_conf = op_conf_util.OperatorConf()
    op_conf.random_like_conf.like = like.logical_blob_name
    if seed is not None:
        op_conf.random_like_conf.random_seed = seed
    setattr(op_conf, "name", name if name is not None else id_util.UniqueStr("RandomLike_"))
    op_conf.random_like_conf.out = "out"
    compile_context.CurJobAddOp(op_conf)
    out_lbi = logical_blob_id_util.LogicalBlobId()
    setattr(out_lbi, "op_name", op_conf.name)
    setattr(out_lbi, "blob_name", "out")
    return remote_blob_util.RemoteBlob(out_lbi)


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
