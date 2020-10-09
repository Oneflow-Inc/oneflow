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
from __future__ import absolute_import
from functools import reduce
from typing import Iterable, List, Optional, Sequence, Union, Tuple
from oneflow.python.oneflow_export import oneflow_export

import numpy as np
import operator
import oneflow as flow
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util
import oneflow.python.framework.interpret_util as interpret_util
import oneflow.python.framework.dtype as dtype_util
import oneflow.python.framework.id_util as id_util
import oneflow.python.framework.remote_blob as remote_blob_util


@oneflow_export("gather")
def gather(
    params: remote_blob_util.BlobDef,
    indices: remote_blob_util.BlobDef,
    validate_indices: Optional[remote_blob_util.BlobDef] = None,
    axis: Optional[int] = None,
    batch_dims: int = 0,
    name: Optional[str] = None,
) -> remote_blob_util.BlobDef:
    r"""Gather slices from params axis axis according to indices.

    Analogous to `tf.gather <https://www.tensorflow.org/api_docs/python/tf/gather>`_

    Args:
        params: A `Blob`. The blob from which to gather values. Must be at least rank `axis + 1`.
        indices: A `Blob`. Index blob. Must be in range [0, params.shape[axis]).
        axis: A `int`. The axis in params to gather indices from. Defaults to the first dimension. 
            Supports negative indexes.
        batch_dims: An optional `int`. Defaults to 0.
        name: A name for the operation (optional).
    Returns:
        A blob. Has the same type as params.
    """
    params_ndims = len(params.shape)
    if axis is None:
        axis = batch_dims
    elif axis < 0:
        origin_axis = axis
        axis += params_ndims
        assert axis >= 0 and axis < params_ndims, ValueError(
            "Expected axis to between [%d, %d).  But received: %d "
            % (-params_ndims, params_ndims, origin_axis)
        )

    if batch_dims > 0:
        if axis == batch_dims:
            return (
                flow.user_op_builder(
                    name if name is not None else id_util.UniqueStr("BatchGather_")
                )
                .Op("batch_gather")
                .Input("in", [params])
                .Input("indices", [indices])
                .Output("out")
                .Build()
                .InferAndTryRun()
                .RemoteBlobList()[0]
            )
        elif axis > batch_dims:
            raise NotImplementedError
        else:
            raise AttributeError
    else:
        return (
            flow.user_op_builder(
                name if name is not None else id_util.UniqueStr("Gather_")
            )
            .Op("gather")
            .Input("in", [params])
            .Input("indices", [indices])
            .Output("out")
            .Attr("axis", int(axis))
            .Build()
            .InferAndTryRun()
            .RemoteBlobList()[0]
        )


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
def reshape(
    x: remote_blob_util.BlobDef, shape: Sequence[int], name: Optional[str] = None
) -> remote_blob_util.BlobDef:
    r"""Reshapes a blob.

    Args:
        x: A `Blob`.
        shape: Shape of the output blob.
        name: A name for the operation (optional).
    Returns:
        A `Blob`, has the same type as `x`.
    """
    x = flow.cast_to_current_logical_view(x)
    assert isinstance(shape, tuple) or isinstance(shape, list)
    shape = list(shape)
    assert all(dim == -1 or dim > 0 for dim in shape)
    assert shape.count(-1) <= 1
    if not x.is_dynamic:
        if name is None:
            name = id_util.UniqueStr("Reshape_")
        return (
            flow.user_op_builder(name)
            .Op("reshape")
            .Input("in", [x])
            .Output("out")
            .Attr("shape", infer_shape(x, shape))
            .Build()
            .InferAndTryRun()
            .RemoteBlobList()[0]
        )
    else:
        op_conf = op_conf_util.OperatorConf()
        setattr(
            op_conf,
            "name",
            name if name is not None else id_util.UniqueStr("DynamicReshape_"),
        )
        setattr(op_conf.dynamic_reshape_conf, "in", x.unique_name)
        op_conf.dynamic_reshape_conf.shape.dim.extend(list(shape))
        setattr(op_conf.dynamic_reshape_conf, "out", "out")
        interpret_util.Forward(op_conf)
        lbi = logical_blob_id_util.LogicalBlobId()
        lbi.op_name = op_conf.name
        lbi.blob_name = "out"
        return remote_blob_util.RemoteBlob(lbi)


@oneflow_export("reshape_like")
def reshape_like(
    x: remote_blob_util.BlobDef,
    like: remote_blob_util.BlobDef,
    name: Optional[str] = None,
) -> remote_blob_util.BlobDef:
    if name is None:
        name = id_util.UniqueStr("ReshapeLike_")
    return (
        flow.user_op_builder(name)
        .Op("reshape_like")
        .Input("in", [x])
        .Input("like", [like])
        .Output("out")
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


@oneflow_export("dynamic_reshape")
def dynamic_reshape(
    x: remote_blob_util.BlobDef, shape: Sequence[int], name: Optional[str] = None
) -> remote_blob_util.BlobDef:
    assert isinstance(shape, tuple) or isinstance(shape, list)
    shape = list(shape)
    op_conf = op_conf_util.OperatorConf()
    setattr(
        op_conf,
        "name",
        name if name is not None else id_util.UniqueStr("DynamicReshape_"),
    )
    setattr(op_conf.dynamic_reshape_conf, "in", x.unique_name)
    op_conf.dynamic_reshape_conf.shape.dim.extend(list(shape))
    setattr(op_conf.dynamic_reshape_conf, "out", "out")
    interpret_util.Forward(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)


@oneflow_export("transpose")
def transpose(
    a: remote_blob_util.BlobDef,
    perm: Sequence[int] = None,
    conjugate: bool = False,
    name: Optional[str] = None,
) -> remote_blob_util.BlobDef:
    r"""Transposes `a`.

    Analogous to `tf.transpose <https://www.tensorflow.org/api_docs/python/tf/transpose>`_

    Args:
        a: A `Blob`.
        perm: A permutation of the dimensions of `a`.
        conjugate: False. Not supported.
        name: A name for the operation (optional).
    Returns:
        A transposed blob.
    """
    assert isinstance(perm, (tuple, list))

    if name is None:
        name = id_util.UniqueStr("Transpose_")

    if conjugate:
        raise NotImplementedError

    return (
        flow.user_op_builder(name)
        .Op("transpose")
        .Input("input", [a])
        .Output("output")
        .Attr("perm", perm)
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


@oneflow_export("slice")
def slice(
    x: remote_blob_util.BlobDef,
    begin: Sequence[int],
    size: Sequence[int],
    name: Optional[str] = None,
) -> remote_blob_util.BlobDef:
    r"""Extracts a slice from a tensor.

    Args:
        x: A `Blob`.
        begin: A list or a tuple, indicate each dimension slice begin, whose length must be equal
            to x's number of dimensions, the first element of beign must be set to None.
            (because oneflow internal slice op do not support slice at dim0 at present)
        size: A list or a tuple, indicate each dimension slice size, whose length must be equal
            to x's number of dimensions, the first element of beign must be set to None.
        name: A name for the operation (optional).
    """
    ndim = len(x.shape)
    if not isinstance(begin, (list, tuple)) or len(begin) != ndim:
        raise ValueError(
            "begin must be a list/tuple with the same length as input tensor's number of dimensions"
        )

    if not all(isinstance(b, int) or b is None for b in begin):
        raise ValueError("element of begin must be a int or None")

    if not isinstance(size, (list, tuple)) or len(size) != ndim:
        raise ValueError(
            "size must be a list/tuple with the same length as input tensor's number of dimensions."
        )

    if not all(isinstance(s, int) or s is None for s in size):
        raise ValueError("element of size must be a int or None")

    slice_tup_list = []
    for b, s, dim_size in zip(begin, size, x.shape):
        start, stop, step = (None, None, 1)
        if b is not None:
            if b < -dim_size or b >= dim_size:
                raise ValueError("element of begin is out of range")
            start = b

        if s is not None:
            if s == -1:
                stop = dim_size
            else:
                if s <= 0 or s > dim_size:
                    raise ValueError("element of size is invalid")
                if b + s < dim_size:
                    stop = b + s

        slice_tup_list.append((start, stop, step))

    return slice_v2(x, slice_tup_list, name=name)


def _check_slice_tup_list(slice_tup_list, shape):
    ndim = len(shape)
    if not isinstance(slice_tup_list, (list, tuple)) or len(slice_tup_list) > ndim:
        raise ValueError(
            "slice_tup_list must be a list or tuple with length "
            "less than or equal to number of dimensions of input tensor"
        )

    # if length of slice_tup_list is less than number of dimensions of x, fill it to length of ndims reduce 1
    if len(slice_tup_list) < ndim:
        slice_tup_list += type(slice_tup_list)(
            [(None, None, None)] * (ndim - len(slice_tup_list))
        )

    start_list = []
    stop_list = []
    step_list = []

    for slice_tup, dim_size in zip(slice_tup_list, shape):
        if not isinstance(slice_tup, (tuple, list)) or len(slice_tup) != 3:
            raise ValueError(
                "element of slice_tup_list must be a list or tuple with form (start, stop, step)"
            )

        if not all(isinstance(idx, int) or idx is None for idx in slice_tup):
            raise ValueError("element of slice tuple must int or None")

        (start, stop, step) = slice_tup
        if step is None:
            step = 1

        if step == 0:
            raise ValueError("slice step can't be 0")

        if start is None:
            start = 0 if step > 0 else np.iinfo(np.int64).max
        elif start < -dim_size or start >= dim_size:
            raise ValueError("slice start must be in range [-size, size)")

        if stop is None:
            stop = np.iinfo(np.int64).max if step > 0 else np.iinfo(np.int64).min
        elif stop < -dim_size - 1 or stop > dim_size:
            raise ValueError("slice start must be in range [-size-1, size]")

        start_list.append(start)
        stop_list.append(stop)
        step_list.append(step)

    return start_list, stop_list, step_list


@oneflow_export("slice_v2")
def slice_v2(
    x: remote_blob_util.BlobDef,
    slice_tup_list: Sequence[Tuple[int, int, int]],
    name: Optional[str] = None,
) -> remote_blob_util.BlobDef:
    r"""Extracts a slice from a tensor.

    Args:
        x: A `Blob`.
        slice_tup_list: A list of slice tuple, indicate each dimension slice (start, stop, step).
        name: A name for the operation (optional).

    """
    name = name or id_util.UniqueStr("Slice_")
    if not isinstance(name, str):
        raise ValueError("name must be a string")

    start, stop, step = _check_slice_tup_list(slice_tup_list, x.shape)

    op = (
        flow.user_op_builder(name)
        .Op("slice")
        .Input("x", [x])
        .Output("y")
        .Attr("start", start)
        .Attr("stop", stop)
        .Attr("step", step)
        .Build()
    )
    return op.InferAndTryRun().SoleOutputBlob()


@oneflow_export("slice_update")
def api_slice_update(
    x: remote_blob_util.BlobDef,
    update: remote_blob_util.BlobDef,
    slice_tup_list: Sequence[Tuple[int, int, int]],
    name: Optional[str] = None,
) -> remote_blob_util.BlobDef:
    r"""Update a slice of tensor `x`.

    Args:
        x: A `Blob`, whose slice will be updated.
        update: A `Blob`, indicate the update content.
        slice_tup_list: A list of slice tuple, indicate each dimension slice (start, stop, step).
        name: A name for the operation (optional).

    """
    if name is None:
        name = id_util.UniqueStr("SliceUpdate_")

    if not isinstance(name, str):
        raise ValueError("name must be a string")

    start, stop, step = _check_slice_tup_list(slice_tup_list, x.shape)

    op = (
        flow.user_op_builder(name)
        .Op("slice_update")
        .Input("x", [x])
        .Input("update", [update])
        .Output("y")
        .Attr("start", start)
        .Attr("stop", stop)
        .Attr("step", step)
        .Build()
    )
    return op.InferAndTryRun().SoleOutputBlob()


@oneflow_export("reverse")
def reverse(
    input: remote_blob_util.BlobDef,
    axis: Union[int, Sequence[int]],
    name: Optional[str] = None,
) -> remote_blob_util.BlobDef:
    if name is None:
        name = id_util.UniqueStr("Reverse_")

    if not isinstance(name, str):
        raise ValueError("name must be a string")

    if isinstance(axis, int):
        axis = [axis]

    if not isinstance(axis, (tuple, list)) or not all(isinstance(a, int) for a in axis):
        raise ValueError("axis must be a int or a list/tuple of int")

    ndim = len(input.shape)
    slice_tup_list = [(None, None, None)] * ndim
    for i, a in enumerate(axis):
        if a < 0:
            a += ndim

        if a < 0 or a >= ndim:
            raise ValueError("axis is out of range")

        slice_tup_list[a] = (None, None, -1)

    return slice_v2(input, slice_tup_list, name)


@oneflow_export("concat")
def concat(
    inputs: Optional[Sequence[remote_blob_util.BlobDef]] = None,
    axis: int = 0,
    max_dim_size: Optional[int] = None,
    name: Optional[str] = None,
    values: Optional[Sequence[remote_blob_util.BlobDef]] = None,
) -> remote_blob_util.BlobDef:
    r"""Concatenate two or more `Blob` s at specified axis.

    Analogous to `numpy.concatenate <https://docs.scipy.org/doc/numpy/reference/generated/numpy.concatenate.html>`_

    Args:
        inputs: a `list` of `Blob`
        axis: a `int`. `0` by default
        max_dim_size: hint of max dimension size along the given axis
        name: name of this operator. `None` by default
        values: deprecated param, use inputs instead

    Returns:
        A `Blob`
    """
    # backward compatible with values param name
    if values is not None:
        assert inputs is None
        inputs = values

    assert isinstance(inputs, (list, tuple))
    if len(inputs) == 1:
        return inputs[0]

    assert len(inputs) >= 2
    if axis < 0:
        axis += len(inputs[0].shape)
    assert axis >= 0 and axis < len(
        inputs[0].shape
    ), "axis must be in range [0, num_axes of inputs)"

    first_input_shape = inputs[0].shape
    static_dim_size = 0
    dynamic_dim_size = 0
    for input in inputs:
        assert len(input.shape) == len(first_input_shape)
        for i in range(len(input.shape)):
            if i == axis:
                if input.is_dynamic:
                    dynamic_dim_size += input.shape[i]
                else:
                    static_dim_size += input.shape[i]
            else:
                assert input.shape[i] == first_input_shape[i]

    if max_dim_size is None:
        max_dim_size = static_dim_size + dynamic_dim_size
    else:
        assert (
            max_dim_size >= static_dim_size
        ), "max diemension size {} is too small to hold concatenated static dimension size {} along the given axis".format(
            max_dim_size, static_dim_size
        )

    if name is None:
        name = id_util.UniqueStr("Concat_")

    op = (
        flow.user_op_builder(name)
        .Op("concat")
        .Input("in", inputs)
        .Output("out")
        .Attr("axis", axis)
        .Attr("max_dim_size", max_dim_size)
        .Build()
    )
    return op.InferAndTryRun().SoleOutputBlob()


@oneflow_export("gather_nd")
def gather_nd(
    params: remote_blob_util.BlobDef,
    indices: remote_blob_util.BlobDef,
    name: Optional[str] = None,
) -> remote_blob_util.BlobDef:
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
    return op.InferAndTryRun().RemoteBlobList()[0]


@oneflow_export("scatter_nd")
def scatter_nd(
    indices: remote_blob_util.BlobDef,
    updates: remote_blob_util.BlobDef,
    shape: Sequence[int],
    name: Optional[str] = None,
):
    if name is None:
        name = id_util.UniqueStr("ScatterNd_")
    op = (
        flow.user_op_builder(name)
        .Op("scatter_nd")
        .Input("indices", [indices])
        .Input("updates", [updates])
        .Attr("shape", shape)
        .Output("out")
        .Build()
    )
    return op.InferAndTryRun().RemoteBlobList()[0]


@oneflow_export("tensor_scatter_nd_update")
def tensor_scatter_nd_update(
    params: remote_blob_util.BlobDef,
    indices: remote_blob_util.BlobDef,
    updates: remote_blob_util.BlobDef,
    name: Optional[str] = None,
) -> remote_blob_util.BlobDef:
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
    return op.InferAndTryRun().RemoteBlobList()[0]


@oneflow_export("tensor_scatter_nd_add")
def tensor_scatter_nd_add(
    params: remote_blob_util.BlobDef,
    indices: remote_blob_util.BlobDef,
    updates: remote_blob_util.BlobDef,
    name: Optional[str] = None,
) -> remote_blob_util.BlobDef:
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
    return op.InferAndTryRun().RemoteBlobList()[0]


@oneflow_export("argwhere")
def argwhere(
    condition: remote_blob_util.BlobDef,
    dtype: Optional[dtype_util.dtype] = None,
    name: Optional[str] = None,
) -> remote_blob_util.BlobDef:
    if name is None:
        name = id_util.UniqueStr("ArgWhere_")

    op_conf = op_conf_util.OperatorConf()
    setattr(op_conf, "name", name)
    setattr(op_conf.arg_where_conf, "in", condition.unique_name)
    setattr(op_conf.arg_where_conf, "out", "out")
    setattr(op_conf.arg_where_conf, "out_size", "out_size")
    if dtype is not None:
        setattr(op_conf.arg_where_conf, "data_type", dtype.oneflow_proto_dtype)
    interpret_util.Forward(op_conf)

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
def nonzero(
    a: remote_blob_util.BlobDef, name: Optional[str] = None
) -> remote_blob_util.BlobDef:
    if name is None:
        argwhere_name = id_util.UniqueStr("Nonzero_ArgWhere_")
        tranpose_name = id_util.UniqueStr("Nonzero_Transpose_")
    else:
        argwhere_name = name + "_ArgWhere"
        tranpose_name = name + "_Transpose"
    indices = argwhere(a, name=argwhere_name)
    return transpose(indices, perm=(1, 0), name=tranpose_name)


@oneflow_export("where")
def where(
    condition: remote_blob_util.BlobDef,
    x: Optional[remote_blob_util.BlobDef] = None,
    y: Optional[remote_blob_util.BlobDef] = None,
    name: Optional[str] = None,
) -> remote_blob_util.BlobDef:
    if x is None and y is None:
        return argwhere(condition, name=name)
    elif x is not None and y is not None:
        if name is None:
            name = id_util.UniqueStr("Where_")

        if x.shape == condition.shape and y.shape == condition.shape:
            broadcast_cond = condition
            broadcast_x = x
            broadcast_y = y
        else:
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
            .InferAndTryRun()
            .RemoteBlobList()[0]
        )
    else:
        raise ValueError("it is not supported when exactly one of x or y is non-None")


@oneflow_export("elem_cnt")
def elem_cnt(
    inputs: remote_blob_util.BlobDef,
    dtype: Optional[dtype_util.dtype] = None,
    name: Optional[str] = None,
) -> remote_blob_util.BlobDef:
    op_conf = op_conf_util.OperatorConf()
    setattr(
        op_conf, "name", name if name is not None else id_util.UniqueStr("ElemCnt_")
    )
    op_conf.shape_elem_cnt_conf.x = inputs.unique_name

    op_conf.shape_elem_cnt_conf.exclude_axis_conf.SetInParent()
    if dtype is not None:
        op_conf.shape_elem_cnt_conf.data_type = dtype.oneflow_proto_dtype
    op_conf.shape_elem_cnt_conf.y = "y"
    interpret_util.Forward(op_conf)
    out_lbi = logical_blob_id_util.LogicalBlobId()
    setattr(out_lbi, "op_name", op_conf.name)
    setattr(out_lbi, "blob_name", "y")
    return remote_blob_util.RemoteBlob(out_lbi)


@oneflow_export("sync_dynamic_resize")
def sync_dynamic_resize(
    inputs: remote_blob_util.BlobDef,
    size: remote_blob_util.BlobDef,
    name: Optional[str] = None,
) -> remote_blob_util.BlobDef:
    op_conf = op_conf_util.OperatorConf()
    setattr(
        op_conf,
        "name",
        name if name is not None else id_util.UniqueStr("SyncDynamicResize_"),
    )
    setattr(op_conf.sync_dynamic_resize_conf, "in", inputs.unique_name)
    setattr(op_conf.sync_dynamic_resize_conf, "size", size.unique_name)
    setattr(op_conf.sync_dynamic_resize_conf, "axis", 0)
    setattr(op_conf.sync_dynamic_resize_conf, "out", "out")
    setattr(op_conf.sync_dynamic_resize_conf, "eager", flow.eager_execution_enabled())
    interpret_util.Forward(op_conf)
    out_lbi = logical_blob_id_util.LogicalBlobId()
    setattr(out_lbi, "op_name", op_conf.name)
    setattr(out_lbi, "blob_name", "out")
    return remote_blob_util.RemoteBlob(out_lbi)


@oneflow_export("stack")
def stack(
    inputs: Sequence[remote_blob_util.BlobDef], axis: int, name: Optional[str] = None
) -> remote_blob_util.BlobDef:
    if not isinstance(inputs, (list, tuple)):
        inputs = [inputs]

    if axis < 0:
        axis = axis + len(inputs[0].shape)

    assert axis == 0, "Only support dim0 stack now."

    op_conf = op_conf_util.OperatorConf()
    setattr(op_conf, "name", name or id_util.UniqueStr("Stack_"))
    getattr(op_conf.stack_conf, "in").extend([input.unique_name for input in inputs])
    setattr(op_conf.stack_conf, "axis", axis)
    setattr(op_conf.stack_conf, "out", "out")
    interpret_util.Forward(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)


@oneflow_export("random.generate_random_batch_permutation_indices")
def generate_random_batch_permutation_indices(
    value: remote_blob_util.BlobDef,
    seed: Optional[int] = None,
    name: Optional[str] = None,
) -> remote_blob_util.BlobDef:
    import random

    op = (
        flow.user_op_builder(
            name
            if name is not None
            else id_util.UniqueStr(value.op_name + "_random_batch_permutation_indices")
        )
        .Op("generate_random_batch_permutation_indices")
        .Input("x", [value])
        .Output("y")
    )
    if seed is not None:
        op.Attr("seed", seed)
        assert name is not None
    else:
        op.Attr("seed", random.randint(-(2 ** 63) + 1, 2 ** 63 - 1))
    return op.Build().InferAndTryRun().RemoteBlobList()[0]


@oneflow_export("random.shuffle")
def shuffle(
    value: remote_blob_util.BlobDef,
    seed: Optional[int] = None,
    name: Optional[str] = None,
) -> remote_blob_util.BlobDef:
    return flow.gather(value, generate_random_batch_permutation_indices(value, seed))


@oneflow_export("identity")
def identity(
    x: remote_blob_util.BlobDef, name: Optional[str] = None
) -> remote_blob_util.BlobDef:
    r"""Return a `Blob` has identical content and data type to input `Blob`. Analogous to `tf.identity <https://www.tensorflow.org/api_docs/python/tf/identity>`_

    Args:
        input: a `Blob`
        name: name of this operator. `None` by default
    
    Returns:
        A `Blob`
    """
    if name is None:
        name = id_util.UniqueStr("Identity_")
    op_conf = op_conf_util.OperatorConf()
    op_conf.name = name
    setattr(op_conf.identity_conf, "in", x.unique_name)
    op_conf.identity_conf.out = "out"
    interpret_util.Forward(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)


@oneflow_export("identity_n")
def identity_n(
    inputs: Iterable[remote_blob_util.BlobDef], name: Optional[str] = None
) -> List[remote_blob_util.BlobDef]:
    op_conf = op_conf_util.OperatorConf()
    setattr(
        op_conf, "name", name if name is not None else id_util.UniqueStr("IdentityN_"),
    )
    assert len(inputs) > 1
    out_bns = []
    for idx, blob in enumerate(inputs):
        getattr(op_conf.tuple_identity_conf, "in").append(blob.unique_name)
        out_bn = "out_" + str(idx)
        getattr(op_conf.tuple_identity_conf, "out").append(out_bn)
        out_bns.append(out_bn)
    interpret_util.Forward(op_conf)

    def bn_to_remote_blob(bn):
        lbi = logical_blob_id_util.LogicalBlobId()
        lbi.op_name = op_conf.name
        lbi.blob_name = bn
        return remote_blob_util.RemoteBlob(lbi)

    return list(map(bn_to_remote_blob, out_bns))


@oneflow_export("squeeze")
def squeeze(
    input: remote_blob_util.BlobDef,
    axis: Optional[Sequence[int]] = None,
    name: Optional[str] = None,
) -> remote_blob_util.BlobDef:
    if axis is None:
        axis = [idx for idx, dim in enumerate(input.shape) if dim == 1]
    else:
        assert isinstance(axis, list) or isinstance(axis, tuple)
        in_num_axes = len(input.shape)
        for x in axis:
            assert x >= -in_num_axes and x < in_num_axes
    return (
        flow.user_op_builder(
            name if name is not None else id_util.UniqueStr("Squeeze_")
        )
        .Op("squeeze")
        .Input("in", [input])
        .Output("out")
        .Attr("axes", list(axis))
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


@oneflow_export("expand_dims")
def expand_dims(
    input: remote_blob_util.BlobDef, axis: int, name: Optional[str] = None
) -> remote_blob_util.BlobDef:
    in_num_axes = len(input.shape)
    assert axis >= -(in_num_axes + 1) and axis <= in_num_axes
    return (
        flow.user_op_builder(
            name if name is not None else id_util.UniqueStr("ExpandDims_")
        )
        .Op("expand_dims")
        .Input("in", [input])
        .Output("out")
        .Attr("axis", axis)
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


@oneflow_export("broadcast_like")
def broadcast_like(
    x: remote_blob_util.BlobDef,
    like: remote_blob_util.BlobDef,
    broadcast_axes: Optional[Sequence[int]] = None,
    name: Optional[str] = None,
) -> remote_blob_util.BlobDef:
    if name is None:
        name = id_util.UniqueStr("BroadcastLike_")

    if broadcast_axes is None:
        broadcast_axes = list(range(len(like.shape)))

    assert isinstance(broadcast_axes, (list, tuple))

    if len(broadcast_axes) <= 0 or len(broadcast_axes) > len(like.shape):
        raise ValueError(
            "The length of broadcast_axes must be greater than 0 and less than or equal to number of axes of like shape"
        )

    op = (
        flow.user_op_builder(name)
        .Op("broadcast_like")
        .Input("x", [x])
        .Input("like", [like])
        .Attr("broadcast_axes", broadcast_axes)
        .Output("y")
        .Build()
    )
    return op.InferAndTryRun().SoleOutputBlob()


@oneflow_export("masked_fill")
def masked_fill(
    x: remote_blob_util.BlobDef,
    mask: remote_blob_util.BlobDef,
    value: Union[float, int],
    name: Optional[str] = None,
) -> remote_blob_util.BlobDef:
    r"""Fill a blob with a given value according to the given mask.

    Args:
        x (remote_blob_util.BlobDef): Input Blob.
        mask (remote_blob_util.BlobDef): Composed with 0 and 1, the input blob 'x' will be 
            filled with the given value where the mask is 1. 
        value (Union[int, int]): The value to use for filling the input blob.
        name (Optional[str], optional): The name for the operation. Defaults to None.
    Attention:
        x and mask must be broadcastable to each other.
        mask must be int type (int8/int32/int64).

    Returns:
        remote_blob_util.BlobDef: The value-filled Blob
    
    For example:

    .. code-block:: python

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp

        @flow.global_function()
        def masked_fill_Job(x: tp.Numpy.Placeholder((4, ), mask: tp.Numpy.Placeholder((4, ),
                            dtype = flow.int8))->tp.Numpy:
            return flow.masked_fill(x, mask, value=5)

        x = np.array([1, 2, 3, 4], dtype=np.float32)
        mask = np.array([1, 0, 0, 1], dtype=np.int8)

        out = masked_fill_Job(x, mask)
        
        # output [5 2 3 5]

    """
    if name is None:
        name = id_util.UniqueStr("MaskedFill_")
    value_like_x = flow.constant_like(like=x, value=value, name=name + "_ConstantLike")
    return flow.where(condition=mask, x=value_like_x, y=x, name=name + "_Where")


@oneflow_export("amp_white_identity")
def amp_white_identity(
    x: remote_blob_util.BlobDef, name: Optional[str] = None
) -> remote_blob_util.BlobDef:
    if name is None:
        name = id_util.UniqueStr("AmpWhiteIdentity_")
    op = (
        flow.user_op_builder(name)
        .Op("amp_white_identity")
        .Input("in", [x])
        .Output("out")
        .Build()
    )
    return op.InferAndTryRun().SoleOutputBlob()
