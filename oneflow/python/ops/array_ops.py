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
import oneflow_api


@oneflow_export("gather")
def gather(
    params: oneflow_api.BlobDesc,
    indices: oneflow_api.BlobDesc,
    validate_indices: Optional[oneflow_api.BlobDesc] = None,
    axis: Optional[int] = None,
    batch_dims: int = 0,
    name: Optional[str] = None,
) -> oneflow_api.BlobDesc:
    r"""This operator gathers slices from params `axis` according to indices.

    Args:
        params: A `Blob`. The blob from which to gather values. Must be at least rank `axis + 1`.
        indices: A `Blob`. Index blob. Must be in range [0, params.shape[axis]).
        axis: A `int`. The axis in params to gather indices from. Defaults to the first dimension.
            Supports negative indexes.
        batch_dims: An optional `int`. Defaults to 0.
        name: A name for the operation (optional).
    Returns:
        A blob. Has the same type as params.

    For example:

    Example 1:

    .. code-block:: python

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def gather_Job(x: tp.Numpy.Placeholder(shape=(3, 3), dtype=flow.float32),
                    indice: tp.Numpy.Placeholder(shape=(2, ), dtype=flow.int32)
        ) -> tp.Numpy:
            gather_blob = flow.gather(params=x,
                                    indices=indice,
                                    axis=1)
            return gather_blob


        x = np.array([[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]]).astype(np.float32)
        indice = np.array([0, 2]).astype(np.int32)
        out = gather_Job(x, indice)

        # out [[1. 3.]
        #      [4. 6.]
        #      [7. 9.]]


    Example 2:

    .. code-block:: python

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def gather_Job(x: tp.Numpy.Placeholder(shape=(3, 3), dtype=flow.float32),
                    indice: tp.Numpy.Placeholder(shape=(2, ), dtype=flow.int32)
        ) -> tp.Numpy:
            gather_blob = flow.gather(params=x,
                                    indices=indice,
                                    axis=0)
            return gather_blob


        x = np.array([[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]]).astype(np.float32)
        indice = np.array([0, 2]).astype(np.int32)
        out = gather_Job(x, indice)

        # out [[1. 2. 3.]
        #      [7. 8. 9.]]

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


@oneflow_export("flatten")
def flatten(
    input: oneflow_api.BlobDesc,
    start_dim: int = 0,
    end_dim: int = -1,
    name: Optional[str] = None,
) -> oneflow_api.BlobDesc:
    r"""Flattens a contiguous range of dims in a Blob.

    Args:
        input: A `Blob`.
        start_dim: The first dim to flatten.
        end_dim: The last dim to flatten.
        name: A name for the operation (optional).
    Returns:
        A `Blob`, has the same type as `input`.

    For example:

    .. code-block:: python

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp

        @flow.global_function()
        def flatten_Job(input: tp.Numpy.Placeholder(shape=(4, 4, 3, 2), dtype=flow.float32)
        ) -> tp.Numpy:
            flatten_blob = flow.flatten(input, start_dim=1, end_dim=-1)
            return flatten_blob


        input = np.zeros((4, 4, 3, 2)).astype(np.float32)
        out = flatten_Job(input)

        # out.shape (4, 24)

    """
    if name is None:
        name = id_util.UniqueStr("Flatten_")
    return (
        flow.user_op_builder(name)
        .Op("flatten")
        .Input("in", [input])
        .Output("out")
        .Attr("start_dim", start_dim)
        .Attr("end_dim", end_dim)
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
    x: oneflow_api.BlobDesc, shape: Sequence[int], name: Optional[str] = None
) -> oneflow_api.BlobDesc:
    r"""This operator reshapes a Blob.
    If the Blob is dynamic, it will call `flow.dynamic_reshape` automatically

    We can set one dimension in `shape` as `-1`, the operator will infer the complete shape.

    Args:
        x: A `Blob`.
        shape: Shape of the output blob.
        name: A name for the operation (optional).
    Returns:
        A `Blob`, has the same type as `x`.

    For example:

    .. code-block:: python

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def reshape_Job(x: tp.Numpy.Placeholder(shape=(4, 4), dtype=flow.float32)
        ) -> tp.Numpy:
            reshape_blob = flow.reshape(x,
                                        shape=[2, 2, 2, -1])
            return reshape_blob


        x = np.array([[1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12],
                    [13, 14, 15, 16]]).astype(np.float32)
        out = reshape_Job(x)

        # out.shape (2, 2, 2, 2)

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
    x: oneflow_api.BlobDesc, like: oneflow_api.BlobDesc, name: Optional[str] = None,
) -> oneflow_api.BlobDesc:
    """This operator reshapes the Blob x to be the same as Blob `like` .

    Args:
        x (oneflow_api.BlobDesc): The input Blob.
        like (oneflow_api.BlobDesc): A Blob.
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow_api.BlobDesc: The result Blob

    For example:

    .. code-block:: python

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def reshape_like_Job(x: tp.Numpy.Placeholder(shape=(4, 4), dtype=flow.float32)
        ) -> tp.Numpy:
            like_blob = flow.constant(value=1,
                                    dtype=flow.int8,
                                    shape=(2, 2, 4))
            reshape_like_blob = flow.reshape_like(x,
                                                like=like_blob)
            return reshape_like_blob


        x = np.array([[1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12],
                    [13, 14, 15, 16]]).astype(np.float32)
        out = reshape_like_Job(x)

        # out.shape (2, 2, 4)

    """
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
    x: oneflow_api.BlobDesc, shape: Sequence[int], name: Optional[str] = None
) -> oneflow_api.BlobDesc:
    """This operator reshapes a dynamic blob.

    Args:
        x (oneflow_api.BlobDesc): The input Blob.
        shape (Sequence[int]): The output shape.
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow_api.BlobDesc: The result Blob.

    For example:

    .. code-block:: python

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp

        @flow.global_function()
        def dynamic_reshape_Job(x: tp.Numpy.Placeholder(shape=(1, 3, 64, 64), dtype=flow.float32)
        ) -> tp.Numpy:
            reshape_out1 = flow.dynamic_reshape(x, (-1, 64))
            variable1 = flow.get_variable(
                "var1",
                shape=(64, 32),
                dtype=flow.float,
                initializer=flow.random_uniform_initializer(minval=-10, maxval=10),
                trainable=True,
            )
            matmul_tensor = flow.matmul(reshape_out1, variable1)
            reshape_out2 = flow.dynamic_reshape(matmul_tensor, (-1, 8, 4))
            return reshape_out2

        x = np.random.rand(1, 3, 64, 64).astype(np.float32)
        out = dynamic_reshape_Job(x)

        # out.shape (192, 8, 4)

    """
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
    a: oneflow_api.BlobDesc,
    perm: Sequence[int] = None,
    conjugate: bool = False,
    batch_axis_non_change: bool = False,
    name: Optional[str] = None,
) -> oneflow_api.BlobDesc:
    r"""This operator transposes the specified axis of input Blob.

    Args:
        a (oneflow_api.BlobDesc): The input Blob.
        perm (Sequence[int], optional): The list of dimension permutation. Defaults to None.
        conjugate (bool, optional): Still Unavailable. Defaults to False.
        batch_axis_non_change (bool, optional): deprecated. Defaults to False.
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Raises:
        NotImplementedError: The attribute `conjugate` still unavailable.

    Returns:
        oneflow_api.BlobDesc: A transposed blob.

    For example:

    .. code-block:: python

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def transpose_Job(x: tp.Numpy.Placeholder(shape=(1, 2, 3), dtype=flow.float32)
        ) -> tp.Numpy:
            transpose_blob = flow.transpose(x,
                                            perm=[2, 0, 1])
            return transpose_blob

        x = np.random.randn(1, 2, 3).astype(np.float32)
        out = transpose_Job(x)

        # out.shape (3, 1, 2)

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
    x: oneflow_api.BlobDesc,
    begin: Sequence[int],
    size: Sequence[int],
    name: Optional[str] = None,
) -> oneflow_api.BlobDesc:
    r"""Extracts a slice from a tensor.

    Args:
        x: A `Blob`.
        begin: A list or a tuple, indicate each dimension slice begin, whose length must be equal
            to x's number of dimensions, the first element of begin must be set to None.
            (Because the internal op of OneFlow does not support 0-dimension slice at present.)
        size: A list or a tuple, indicate each dimension slice size, whose length must be equal
            to x's number of dimensions, the first element of beign must be set to None.
        name: A name for the operation (optional).

    Returns:
        oneflow_api.BlobDesc: The result Blob.

    For example:

    .. code-block:: python

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def slice_Job(x: tp.Numpy.Placeholder(shape=(3, 3), dtype=flow.float32)
        ) -> tp.Numpy:
            slice_blob = flow.slice(x,
                                    begin=[None, 0],
                                    size=[None, 2])
            return slice_blob

        x = np.array([[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]]).astype(np.float32)
        out = slice_Job(x)

        # out [[1. 2.]
        #      [4. 5.]
        #      [7. 8.]]

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
    x: oneflow_api.BlobDesc,
    slice_tup_list: Sequence[Tuple[int, int, int]],
    name: Optional[str] = None,
) -> oneflow_api.BlobDesc:
    r"""Extracts a slice from a tensor.
    The `slice_tup_list` assigns the slice indices in each dimension, the format is (start, stop, step).
    The operator will slice the Blob according to the `slice_top_list`.

    Args:
        x: A `Blob`.
        slice_tup_list: A list of slice tuple, indicate each dimension slice (start, stop, step).
        name: A name for the operation (optional).

    Returns:
        oneflow_api.BlobDesc: The result Blob.

    Note: Because the internal op of OneFlow does not support 0-dimension slice at present, we should
    set the zero element in `slice_tup_list` as `None`.

    For example:

    .. code-block:: python

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp
        @flow.global_function()
        def slicev2_Job(x: tp.Numpy.Placeholder(shape=(3, 6, 9), dtype=flow.float32)
        ) -> tp.Numpy:
            slicev2_blob = flow.slice_v2(x,
                                        slice_tup_list=[[None, None, None],
                                                        [0, 5, 2], # slice in dimension 1, extract [0, 2, 4]
                                                        [0, 6, 3]]) # slice in dimension 2, extract [0, 3]
            return slicev2_blob
        x = np.random.randn(3, 6, 9).astype(np.float32)
        out = slicev2_Job(x)

        # out.shape (3, 3, 2)

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
    x: oneflow_api.BlobDesc,
    update: oneflow_api.BlobDesc,
    slice_tup_list: Sequence[Tuple[int, int, int]],
    name: Optional[str] = None,
) -> oneflow_api.BlobDesc:
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


# Get slice attrs for slice_assign and logical_slice
# Note the step in slice_tup_list must be greater than 0
# as slice_assign and logical_slice only support step > 0
def _GetSliceAttrs(slice_tup_list, input_shape):
    ndim = len(input_shape)
    if not (isinstance(slice_tup_list, (list, tuple)) and len(slice_tup_list) <= ndim):
        raise ValueError(
            "slice_tup_list must be a list or tuple with length "
            "less than or equal to number of dimensions of input tensor"
        )

    # Right extends slice_tup_list with [None, None, None] if len(slice_tup_list) < len(input_shape)
    if len(slice_tup_list) < ndim:
        slice_tup_list += type(slice_tup_list)(
            [(None, None, None)] * (ndim - len(slice_tup_list))
        )

    start_list = []
    stop_list = []
    step_list = []

    for slice_tup, dim_size in zip(slice_tup_list, input_shape):
        if not (isinstance(slice_tup, (tuple, list)) and len(slice_tup) == 3):
            raise ValueError(
                "element of slice_tup_list must be a list or tuple with form (start, stop, step)"
            )

        if not all(isinstance(idx, int) or idx is None for idx in slice_tup):
            raise ValueError("element of slice tuple must int or None")

        (start, stop, step) = slice_tup
        if step is None:
            step = 1

        if step <= 0:
            raise ValueError("slice_assign/logical_slice step must be greater than 0")

        if start is None:
            start = 0
        elif start < -dim_size or start >= dim_size:
            raise ValueError(
                "slice_assign/logical_slice start must be in range [-size, size)"
            )
        elif start < 0:
            start += dim_size

        if stop is None:
            stop = dim_size
        elif stop < -dim_size or stop > dim_size:
            raise ValueError(
                "slice_assign/logical_slice start must be in range [-size, size]"
            )
        elif stop < 0:
            stop += dim_size

        start_list.append(start)
        stop_list.append(stop)
        step_list.append(step)

    return start_list, stop_list, step_list


@oneflow_export("experimental.logical_slice")
def logical_slice(
    x: oneflow_api.BlobDesc,
    slice_tup_list: Sequence[Tuple[int, int, int]],
    name: Optional[str] = None,
) -> oneflow_api.BlobDesc:

    name = id_util.UniqueStr("LogicalSlice_") if name is None else name
    if not isinstance(name, str):
        raise ValueError("name must be a string")

    start_list, stop_list, step_list = _GetSliceAttrs(slice_tup_list, x.shape)
    op = (
        flow.user_op_builder(name)
        .Op("logical_slice")
        .Input("x", [x])
        .Output("y")
        .Attr("start", start_list)
        .Attr("stop", stop_list)
        .Attr("step", step_list)
        .Build()
    )
    return op.InferAndTryRun().SoleOutputBlob()


@oneflow_export("experimental.logical_slice_assign")
def logical_slice_assign(
    x: oneflow_api.BlobDesc,
    value: oneflow_api.BlobDesc,
    slice_tup_list: Sequence[Tuple[int, int, int]],
    name: Optional[str] = None,
) -> oneflow_api.BlobDesc:

    name = id_util.UniqueStr("LogicalSliceAssign_") if name is None else name
    if not isinstance(name, str):
        raise ValueError("name must be a string")

    start_list, stop_list, step_list = _GetSliceAttrs(slice_tup_list, x.shape)
    op = (
        flow.user_op_builder(name)
        .Op("logical_slice_assign")
        .Input("ref", [x])
        .Input("value", [value])
        .Attr("start", start_list)
        .Attr("stop", stop_list)
        .Attr("step", step_list)
        .Build()
    )
    return op.InferAndTryRun()


@oneflow_export("reverse")
def reverse(
    input: oneflow_api.BlobDesc,
    axis: Union[int, Sequence[int]],
    name: Optional[str] = None,
) -> oneflow_api.BlobDesc:
    """This operator reverses the elements on the assigned axis.

    Args:
        input (oneflow_api.BlobDesc): The input Blob.
        axis (Union[int, Sequence[int]]): The reverse axis.
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Raises:
        ValueError: The name must be a string.
        ValueError: The axis must be a int or a list/tuple of int.
        ValueError: The axis is out of range.

    Returns:
        oneflow_api.BlobDesc: The result Blob

    For example:

    .. code-block:: python

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def reverse_Job(x: tp.Numpy.Placeholder(shape=(3, 3), dtype=flow.float32)) -> tp.Numpy:
            reverse_blob = flow.reverse(x,
                                        axis=0)
            return reverse_blob


        x = np.array([[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]]).astype(np.float32)
        out = reverse_Job(x)

        # out [[7. 8. 9.]
        #      [4. 5. 6.]
        #      [1. 2. 3.]]

    """
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
    inputs: Optional[Sequence[oneflow_api.BlobDesc]] = None,
    axis: int = 0,
    max_dim_size: Optional[int] = None,
    name: Optional[str] = None,
    values: Optional[Sequence[oneflow_api.BlobDesc]] = None,
) -> oneflow_api.BlobDesc:
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

    For example:

    .. code-block:: python

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def concat_Job() -> tp.Numpy:
            constant_blob_1 = flow.constant(value=1.5,
                                            shape=(1, 3, 3, 4),
                                            dtype=flow.float,
                                            name="blob1")
            constant_blob_2 = flow.constant(value=2.5,
                                            shape=(1, 3, 3, 4),
                                            dtype=flow.float,
                                            name="blob2")
            return flow.concat(inputs=[constant_blob_1, constant_blob_2],
                            axis=3)


        out = concat_Job()

        # out.shape (1, 3, 3, 8)

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
    params: oneflow_api.BlobDesc,
    indices: oneflow_api.BlobDesc,
    name: Optional[str] = None,
) -> oneflow_api.BlobDesc:
    """This operator is a high-dimensional extension of `gather`, `indices` is a K-dimensional
    tensor, which is regarded as a index of input Blob `params`.

    Each element defines a slice of `params`:

    .. math::

        output[(i_0,i_1,...,i_{K-2})] = param[indices(i_{0},i_{1},...,i_{K-2})]


    Args:
        params (oneflow_api.BlobDesc): The input Blob.
        indices (oneflow_api.BlobDesc): The slice indices.
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow_api.BlobDesc: The result Blob.

    For example:

    Example 1:

    .. code-block:: python

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def gather_nd_Job(x: tp.Numpy.Placeholder(shape=(3, 3), dtype=flow.float32),
                        indice: tp.Numpy.Placeholder(shape=(2, 1), dtype=flow.int32)
        ) -> tp.Numpy:
            gather_nd_blob = flow.gather_nd(params=x,
                                            indices=indice)
            return gather_nd_blob


        x = np.array([[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]]).astype(np.float32)
        indice = np.array([[0], [2]]).astype(np.int32)
        out = gather_nd_Job(x, indice)

        # out [[1. 2. 3.]
        #      [7. 8. 9.]]

    Example 2:

    .. code-block:: python

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def gather_nd_Job(x: tp.Numpy.Placeholder(shape=(3, 3), dtype=flow.float32),
                        indice: tp.Numpy.Placeholder(shape=(2, 2), dtype=flow.int32)
        ) -> tp.Numpy:
            gather_nd_blob = flow.gather_nd(params=x,
                                            indices=indice)
            return gather_nd_blob


        x = np.array([[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]]).astype(np.float32)
        indice = np.array([[0, 2], [2, 1]]).astype(np.int32)
        out = gather_nd_Job(x, indice)

        # out [3. 8.]

    Example3:

    .. code-block:: python

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def gather_nd_Job(x: tp.Numpy.Placeholder(shape=(3, 3), dtype=flow.float32),
                        indice: tp.Numpy.Placeholder(shape=(3, 2), dtype=flow.int32)
        ) -> tp.Numpy:
            gather_nd_blob = flow.gather_nd(params=x,
                                            indices=indice)
            return gather_nd_blob


        x = np.array([[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]]).astype(np.float32)
        indice = np.array([[0, 1], [1, 0], [2, 2]]).astype(np.int32)
        out = gather_nd_Job(x, indice)

        # out [2. 4. 9.]

    """
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
    indices: oneflow_api.BlobDesc,
    updates: oneflow_api.BlobDesc,
    shape: Sequence[int],
    name: Optional[str] = None,
):
    """This operator inserts the elements in `updates` according to the `indices` and create a new Blob.

    Args:
        indices (oneflow_api.BlobDesc): The indice of `updates`. Its type should be `flow.int`.
        updates (oneflow_api.BlobDesc): The update Blob.
        shape (Sequence[int]): The constant tensor shape, the constant tensor elements are all zero.
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow_api.BlobDesc: The result Blob.

    For example:

    Example 1:

    .. code-block:: python

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def scatter_nd_Job(indice: tp.Numpy.Placeholder(shape=(3, 1), dtype=flow.int32),
                        update: tp.Numpy.Placeholder(shape=(3, ), dtype=flow.float32),
        ) -> tp.Numpy:
            scatter_blob = flow.scatter_nd(indices=indice,
                                        updates=update,
                                        shape=[8])
            return scatter_blob


        indice_array = np.array([[1], [6], [4]]).astype(np.int32)
        update_array = np.array([10.2, 5.1, 12.7]).astype(np.float32)
        out = scatter_nd_Job(indice_array, update_array)

        # [ 0.  10.2  0.   0.  12.7  0.   5.1  0. ]

    Example 2:

    .. code-block:: python

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def scatter_nd_Job(indice: tp.Numpy.Placeholder(shape=(3, 1), dtype=flow.int32),
                        update: tp.Numpy.Placeholder(shape=(3, 3), dtype=flow.float32),
        ) -> tp.Numpy:
            scatter_blob = flow.scatter_nd(indices=indice,
                                        updates=update,
                                        shape=[5, 3])
            return scatter_blob


        indice_array = np.array([[0], [4], [2]]).astype(np.int32)
        update_array = np.array([[1, 1, 1],
                                [2, 2, 2],
                                [3, 3, 3]]).astype(np.float32)
        out = scatter_nd_Job(indice_array, update_array)

        # out [[1. 1. 1.]
        #      [0. 0. 0.]
        #      [3. 3. 3.]
        #      [0. 0. 0.]
        #      [2. 2. 2.]]

    """
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
    params: oneflow_api.BlobDesc,
    indices: oneflow_api.BlobDesc,
    updates: oneflow_api.BlobDesc,
    name: Optional[str] = None,
) -> oneflow_api.BlobDesc:
    """This operator inserts the elements in `updates` according to the `indices` into the Blob `params`.

    Args:
        params (oneflow_api.BlobDesc): The input Blob.
        indices (oneflow_api.BlobDesc): The indice of `updates`. Its type should be `flow.int32`.
        updates (oneflow_api.BlobDesc): The update Blob.
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow_api.BlobDesc: The result Blob.

    For example:

    .. code-block:: python

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def tensor_scatter_nd_Job(x: tp.Numpy.Placeholder(shape=(5, 3), dtype=flow.float32),
                                indice: tp.Numpy.Placeholder(shape=(3, 1), dtype=flow.int32),
                                update: tp.Numpy.Placeholder(shape=(3, 3), dtype=flow.float32),
        ) -> tp.Numpy:
            scatter_blob = flow.tensor_scatter_nd_update(params=x,
                                                        indices=indice,
                                                        updates=update)
            return scatter_blob

        x = np.array([[1, 2, 3],
                    [1, 2, 3],
                    [1, 2, 3],
                    [1, 2, 3],
                    [1, 2, 3]]).astype(np.float32)
        indice_array = np.array([[0], [4], [2]]).astype(np.int32)
        update_array = np.array([[1, 1, 1],
                                [2, 2, 2],
                                [3, 3, 3]]).astype(np.float32)
        out = tensor_scatter_nd_Job(x, indice_array, update_array)

        # out [[1. 1. 1.]
        #      [1. 2. 3.]
        #      [3. 3. 3.]
        #      [1. 2. 3.]
        #      [2. 2. 2.]]

    """
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
    params: oneflow_api.BlobDesc,
    indices: oneflow_api.BlobDesc,
    updates: oneflow_api.BlobDesc,
    name: Optional[str] = None,
) -> oneflow_api.BlobDesc:
    """This operator adds elements from 'updates' to Blob 'params' based on the `indices`.

    Args:
        params (oneflow_api.BlobDesc): The input Blob.
        indices (oneflow_api.BlobDesc): The indice of `updates`. Its type should be `flow.int32`.
        updates (oneflow_api.BlobDesc): The update Blob.
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow_api.BlobDesc: The result Blob.

    For example：

    .. code-block:: python

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def tensor_scatter_nd_add_Job(x: tp.Numpy.Placeholder(shape=(5, 3), dtype=flow.float32),
                                    indice: tp.Numpy.Placeholder(shape=(3, 1), dtype=flow.int32),
                                    update: tp.Numpy.Placeholder(shape=(3, 3), dtype=flow.float32),
        ) -> tp.Numpy:
            scatter_blob = flow.tensor_scatter_nd_add(params=x,
                                                    indices=indice,
                                                    updates=update)
            return scatter_blob

        x = np.array([[1, 2, 3],
                    [1, 2, 3],
                    [1, 2, 3],
                    [1, 2, 3],
                    [1, 2, 3]]).astype(np.float32)
        indice_array = np.array([[0], [4], [2]]).astype(np.int32)
        update_array = np.array([[1, 1, 1],
                                [2, 2, 2],
                                [3, 3, 3]]).astype(np.float32)
        out = tensor_scatter_nd_add_Job(x, indice_array, update_array)

        # out [[2. 3. 4.]
        #      [1. 2. 3.]
        #      [4. 5. 6.]
        #      [1. 2. 3.]
        #      [3. 4. 5.]]

    """
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
    condition: oneflow_api.BlobDesc,
    dtype: Optional[dtype_util.dtype] = None,
    name: Optional[str] = None,
) -> oneflow_api.BlobDesc:
    """This operator finds the indices of input Blob `condition` elements that are non-zero. It returns a List.
    Each element in the output is a coordinate that points to a non-zero element in the condition.

    Args:
        condition (oneflow_api.BlobDesc): The input Blob.
        dtype (Optional[dtype_util.dtype], optional): The data type of output. Defaults to None.
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow_api.BlobDesc: The result Blob. Its type is `ListNumpy`.

    For example:

    .. code-block:: python

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def argwhere_Job(x: tp.Numpy.Placeholder(shape=(2, 3), dtype=flow.float32),
        ) -> tp.ListNumpy:
            return flow.argwhere(x)


        x = np.array([[0, 1, 0],
                    [2, 0, 2]]).astype(np.float32)
        out = argwhere_Job(x)

        # out [array([[0, 1],
        #             [1, 0],
        #             [1, 2]], dtype=int32)]

    """
    if name is None:
        name = id_util.UniqueStr("ArgWhere_")

    if dtype is None:
        dtype = flow.int32

    op = (
        flow.user_op_builder(name)
        .Op("argwhere")
        .Input("input", [condition])
        .Attr("dtype", dtype)
        .Output("output")
        .Output("output_size")
        .Build()
    )
    output, output_size = op.InferAndTryRun().RemoteBlobList()
    return sync_dynamic_resize(output, output_size)


@oneflow_export("nonzero")
def nonzero(
    a: oneflow_api.BlobDesc, name: Optional[str] = None
) -> oneflow_api.BlobDesc:
    """This operator finds the indices of input Blob `condition` elements that are non-zero.

    Args:
        a (oneflow_api.BlobDesc): The input Blob.
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow_api.BlobDesc: The result Blob.
    """
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
    condition: oneflow_api.BlobDesc,
    x: Optional[oneflow_api.BlobDesc] = None,
    y: Optional[oneflow_api.BlobDesc] = None,
    name: Optional[str] = None,
) -> oneflow_api.BlobDesc:
    """This operator returns the elements where condition is larger than 0.

    If `x` and `y` is None, this operator is equal to `oneflow.argwhere`.

    If `x` and `y` both are not None, If the element in condition is larger than 0,
    it will take the `x` element, else it will take the `y` element.

    Args:
        condition (oneflow_api.BlobDesc): The input Blob.
        x (Optional[oneflow_api.BlobDesc], optional): A Blob. Defaults to None.
        y (Optional[oneflow_api.BlobDesc], optional): A Blob. Defaults to None.
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Raises:
        ValueError: It is not supported when exactly one of x or y is non-None

    Returns:
        oneflow_api.BlobDesc: The result Blob. Its type is `ListNumpy`.

    For example:

    Example 1:

    .. code-block:: python

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def where_Job(condition: tp.Numpy.Placeholder(shape=(5, ), dtype=flow.int32),
                    x: tp.Numpy.Placeholder(shape=(5, ), dtype=flow.float32),
                    y: tp.Numpy.Placeholder(shape=(5, ), dtype=flow.float32),
        ) -> tp.ListNumpy:
            return flow.where(condition=condition,
                            x=x,
                            y=y)


        condition = np.array([3, 0, 1, 0, 1]).astype(np.int32)
        x = np.array([10, 20, 30, 40, 50]).astype(np.float32)
        y = np.array([100, 200, 300, 400, 500]).astype(np.float32)
        out = where_Job(condition, x, y)

        # out [array([ 10., 200.,  30., 400.,  50.], dtype=float32)]

    Example 2:

    .. code-block:: python

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def where_Job(condition: tp.Numpy.Placeholder(shape=(5, ), dtype=flow.int32),
        ) -> tp.ListNumpy:
            return flow.where(condition=condition)


        condition = np.array([3, 0, 1, 0, 1]).astype(np.int32)
        out = where_Job(condition)

        # out [array([[0],
        #             [2],
        #             [4]], dtype=int32)]

    """
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
    inputs: oneflow_api.BlobDesc,
    dtype: Optional[dtype_util.dtype] = None,
    name: Optional[str] = None,
) -> oneflow_api.BlobDesc:
    """This operator returns the amount of elements in input Blob.

    Args:
        inputs (oneflow_api.BlobDesc): The input Blob.
        dtype (Optional[dtype_util.dtype], optional): The data type. Defaults to None.
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow_api.BlobDesc: The result Blob. Its type is `ListNumpy`.

    For example:

    .. code-block:: python

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def elem_cnt_Job(x: tp.Numpy.Placeholder(shape=(5, ), dtype=flow.float32),
        ) -> tp.ListNumpy:
            return flow.elem_cnt(inputs=x, dtype=flow.int32)

        x = np.array([10, 20, -30, 40, 50]).astype(np.float32)
        out = elem_cnt_Job(x)

        # [array([5], dtype=int32)]

    """
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
    inputs: oneflow_api.BlobDesc,
    size: oneflow_api.BlobDesc,
    name: Optional[str] = None,
) -> oneflow_api.BlobDesc:
    """

    Args:
        inputs (oneflow_api.BlobDesc): The input Blob.
        size (oneflow_api.BlobDesc): The size of new Blob.
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow_api.BlobDesc: The result Blob. Its type is `ListNumpy`.

    For example:

    .. code-block:: python

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def sync_dynamic_resize_Job(x: tp.Numpy.Placeholder(shape=(4, 3), dtype=flow.float32),
                                    size: tp.Numpy.Placeholder(shape=(1, ), dtype=flow.int32),
        ) -> tp.ListNumpy:
            resize_Blob = flow.sync_dynamic_resize(inputs=x,
                                                size=size)
            return resize_Blob

        x = np.array([[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9],
                    [10, 11, 12]]).astype(np.float32)
        size = np.array([2]).astype(np.int32)
        out = sync_dynamic_resize_Job(x, size)

        # out [array([[1., 2., 3.],
        #             [4., 5., 6.]], dtype=float32)]

    """
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
    inputs: Sequence[oneflow_api.BlobDesc], axis: int = 0, name: Optional[str] = None,
) -> oneflow_api.BlobDesc:
    """This operator stacks the multiple Blobs on the specified axis.

    Args:
        inputs (Sequence[oneflow_api.BlobDesc]): A list of input Blob.
        axis (int): The stack axis.
        name (Optional[str], optional): The name for the operation. Defaults to None.

    For example:

    .. code-block:: python

        import oneflow as flow
        import oneflow.typing as tp
        import numpy as np


        @flow.global_function()
        def stack_job(x: tp.Numpy.Placeholder(shape=(2, 4, 6)),
                    y: tp.Numpy.Placeholder(shape=(2, 4, 6)))->tp.Numpy:
            out = flow.stack([x, y], axis=2)
            return out

        x = np.ones(shape=(2, 4, 6), dtype=np.float32)
        y = np.ones(shape=(2, 4, 6), dtype=np.float32)

        out = stack_job(x, y)

        # output.shape (2, 4, 2, 6)

    Returns:
        oneflow_api.BlobDesc: The result Blob.

    """
    if name is None:
        name = id_util.UniqueStr("Stack_")

    inputs = list(inputs)

    _input_shape = inputs[0].shape
    _max_dim = len(_input_shape)

    # The axis must be in range [-(_max_dim +1), _max_dim]
    if axis < 0:
        axis = axis + _max_dim + 1
    assert (axis >= 0) and (axis <= _max_dim)

    # All input tensors must have the same shape
    _input_list_length = len(inputs)
    for i in range(_input_list_length):
        _current_shape = inputs[i].shape
        assert (
            _input_shape == _current_shape
        ), "Each tensor should have the same shape ! Found a tensor instance shape is: {}".format(
            _current_shape
        )
        # Expand dims for each tensor
        inputs[i] = flow.expand_dims(
            inputs[i], axis=axis, name=name + "expand_dims_{}".format(i)
        )

    return flow.concat(inputs, axis=axis, name=name + "concat")


@oneflow_export("random.generate_random_batch_permutation_indices")
def generate_random_batch_permutation_indices(
    value: oneflow_api.BlobDesc, seed: Optional[int] = None, name: Optional[str] = None,
) -> oneflow_api.BlobDesc:
    """This operator generates a random permutation of indices in batch axis.

    Args:
        value (oneflow_api.BlobDesc): The input Blob.
        seed (Optional[int], optional): The random seed. Defaults to None.
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow_api.BlobDesc: The result Blob. Its type is `ListNumpy`.

    For example:

    .. code-block:: python

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def random_indice_Job(x: tp.Numpy.Placeholder(shape=(4, 3), dtype=flow.int32),
        ) -> tp.ListNumpy:
            return flow.random.generate_random_batch_permutation_indices(value=x)

        x = np.array([[1, 1, 1],
                    [2, 2, 2],
                    [3, 3, 3],
                    [4, 4, 4]]).astype(np.int32)
        out = random_indice_Job(x)

        # out [array([3, 0, 2, 1], dtype=int32)]

    """
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
    value: oneflow_api.BlobDesc, seed: Optional[int] = None, name: Optional[str] = None,
) -> oneflow_api.BlobDesc:
    """This operator shuffle the elements in input Blob.

    Args:
        value (oneflow_api.BlobDesc): The input Blob.
        seed (Optional[int], optional): The random seed. Defaults to None.
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow_api.BlobDesc: The result Blob.

    For example:

    .. code-block:: python

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def shuffle_Job(x: tp.Numpy.Placeholder(shape=(3, 3), dtype=flow.int32),
        ) -> tp.Numpy:
            return flow.random.shuffle(x)

        x = np.array([[1, 1, 1],
                    [2, 2, 2],
                    [3, 3, 3]]).astype(np.int32)
        out = shuffle_Job(x)

        # out [[3 3 3]
        #      [1 1 1]
        #      [2 2 2]]

    """
    return flow.gather(value, generate_random_batch_permutation_indices(value, seed))


@oneflow_export("identity")
def identity(
    x: oneflow_api.BlobDesc, name: Optional[str] = None
) -> oneflow_api.BlobDesc:
    r"""This operator returns a `Blob` that has identical content and data type to input `Blob`.

    Analogous to `tf.identity <https://www.tensorflow.org/api_docs/python/tf/identity>`_

    Args:
        x (oneflow_api.BlobDesc): The input Blob.
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow_api.BlobDesc: The result Blob.

    For example:

    .. code-block:: python

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def identity_Job(x: tp.Numpy.Placeholder(shape=(3, 3), dtype=flow.int32),
        ) -> tp.Numpy:
            return flow.identity(x)

        x = np.array([[1, 1, 1],
                    [2, 2, 2],
                    [3, 3, 3]]).astype(np.int32)
        out = identity_Job(x)

        # out [[1 1 1]
        #      [2 2 2]
        #      [3 3 3]]

    """
    if name is None:
        name = id_util.UniqueStr("Identity_")

    op = (
        flow.user_op_builder(name).Op("identity").Input("in", [x]).Output("out").Build()
    )
    return op.InferAndTryRun().SoleOutputBlob()


@oneflow_export("identity_n")
def identity_n(
    inputs: Sequence[oneflow_api.BlobDesc], name: Optional[str] = None
) -> List[oneflow_api.BlobDesc]:
    """This operator is similar to `oneflow.identity`. The difference is that the input and output
    of `identity_n` is `List`.

    Args:
        inputs (Iterable[oneflow_api.BlobDesc]): A List of input Blob.
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        List[oneflow_api.BlobDesc]: A list of result Blob.

    For example:

    .. code-block:: python

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp
        from typing import List


        @flow.global_function()
        def identity_Job(x: tp.Numpy.Placeholder(shape=(1, 3), dtype=flow.int32),
                        y: tp.Numpy.Placeholder(shape=(1, 3), dtype=flow.int32),
                        z: tp.Numpy.Placeholder(shape=(1, 3), dtype=flow.int32)
        ) -> List[tp.Numpy]:
            return flow.identity_n([x, y, z])


        x = np.array([[1, 1, 1]]).astype(np.int32)
        y = np.array([[2, 2, 2]]).astype(np.int32)
        z = np.array([[3, 3, 3]]).astype(np.int32)
        out = identity_Job(x, y, z)

        # out[0] [[1, 1, 1]]
        # out[1] [[2, 2, 2]]
        # out[2] [[3, 3, 3]]

    """
    return (
        flow.user_op_builder(
            name if name is not None else id_util.UniqueStr("IdentityN_")
        )
        .Op("tuple_identity")
        .Input("in", inputs)
        .Output("out", len(inputs))
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()
    )


@oneflow_export("cast_to_static_shape")
def cast_to_static_shape(
    x: oneflow_api.BlobDesc, name: Optional[str] = None
) -> oneflow_api.BlobDesc:
    r"""This operator returns a `Blob` that has identical content and data type to input `Blob`, and whose shape is converted from dynamic to static

    Args:
        x (oneflow_api.BlobDesc): The input Blob which has dynamic shape.
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow_api.BlobDesc: The result Blob which is identical to input blob but has static shape.

    For example:

    .. code-block:: python

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp

        @flow.global_function()
        def cast_to_static_shape_func(
            x: tp.ListNumpy.Placeholder(shape=(3, 3), dtype=flow.float32),
        ) -> tp.Numpy:
            return flow.cast_to_static_shape(x)

        x = np.array([[1, 1, 1],
                      [2, 2, 2],
                      [3, 3, 3]]).astype(np.float32)

        out = cast_to_static_shape_func(x)

        # out [[1 1 1]
        #      [2 2 2]
        #      [3 3 3]]

    """
    if not x.is_dynamic:
        return x

    if name is None:
        name = id_util.UniqueStr("CastToStaticShape_")

    op = (
        flow.user_op_builder(name)
        .Op("cast_to_static_shape")
        .Input("input", [x])
        .Output("output")
        .Build()
    )
    return op.InferAndTryRun().SoleOutputBlob()


@oneflow_export("squeeze")
def squeeze(
    input: oneflow_api.BlobDesc,
    axis: Optional[Sequence[int]] = None,
    name: Optional[str] = None,
) -> oneflow_api.BlobDesc:
    """This operator removes the specified dimention which size is 1 of the input Blob.
    If the `axis` is not specified, this operator will remove all the dimention which size is 1 of the input Blob.

    The amount of element in return value is the same as Blob `input`.

    Args:
        input (oneflow_api.BlobDesc): The input Blob.
        axis (Optional[Sequence[int]], optional): The axis. Defaults to None.
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow_api.BlobDesc: The result Blob.

    For example:

    Example 1:

    .. code-block:

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def squeeze_Job(x: tp.Numpy.Placeholder(shape=(1, 1, 1, 3), dtype=flow.int32),
        ) -> tp.Numpy:
            return flow.squeeze(x)


        x = np.array([[[[1, 1, 1]]]]).astype(np.int32)
        out = squeeze_Job(x)

        # out.shape (3,)

    Example 2:

    .. code-block:: python

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def squeeze_Job(x: tp.Numpy.Placeholder(shape=(1, 1, 1, 3), dtype=flow.int32),
        ) -> tp.Numpy:
            return flow.squeeze(x, axis=[1, 2])


        x = np.array([[[[1, 1, 1]]]]).astype(np.int32)
        out = squeeze_Job(x)

        # out.shape (1, 3)

    """
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
    input: oneflow_api.BlobDesc, axis: int, name: Optional[str] = None
) -> oneflow_api.BlobDesc:
    """This operator inserts a dimention at the specified axis in the input Blob.
    The size of new dimension can only be 1, and the amount of element in return value is the same as Blob `input`.

    Args:
        input (oneflow_api.BlobDesc): The input Blob.
        axis (int): The specified dimension index.
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow_api.BlobDesc: The result Blob.

    For example:

    .. code-block:: python

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def expand_dim_Job(x: tp.Numpy.Placeholder(shape=(1, 3, 3), dtype=flow.int32),
        ) -> tp.Numpy:
            return flow.expand_dims(input=x,
                                    axis=2)


        x = np.array([[[1, 1, 1],
                    [1, 1, 1],
                    [1, 1, 1]]]).astype(np.int32)
        out = expand_dim_Job(x)

        # out.shape (1, 3, 1, 3)

    """
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
    x: oneflow_api.BlobDesc,
    like: oneflow_api.BlobDesc,
    broadcast_axes: Optional[Sequence[int]] = None,
    name: Optional[str] = None,
) -> oneflow_api.BlobDesc:
    """This operator broadcast the input Blob `x` on the specified axis with input Blob `like`.

    Args:
        x (oneflow_api.BlobDesc): The input Blob.
        like (oneflow_api.BlobDesc): A Blob.
        broadcast_axes (Optional[Sequence[int]], optional): The broadcast axis. Defaults to None.
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Raises:
        ValueError: The length of broadcast_axes must be greater than 0 and less than or equal to number of axes of like shape.

    Returns:
        oneflow_api.BlobDesc: The result Blob.

    For example:

    Example 1:

    .. code-block:: python

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def broadcast_like_Job(x: tp.Numpy.Placeholder(shape=(3, 1), dtype=flow.float32)
        ) -> tp.Numpy:
            like_tensor = flow.constant(value=1.0,
                                        dtype=flow.float32,
                                        shape=(3, 3))
            return flow.broadcast_like(x=x,
                                    like=like_tensor,
                                    broadcast_axes=(1, ))


        x = np.array([[1], [1], [1]]).astype(np.float32)
        out = broadcast_like_Job(x)

        # out [[[1 1 1]
        #       [1 1 1]
        #       [1 1 1]]]

        # out.shape (3, 3)

    Example 2:

    .. code-block:: python

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def broadcast_like_Job(x: tp.Numpy.Placeholder(shape=(3, 1, 1), dtype=flow.float32)
        ) -> tp.Numpy:
            like_tensor = flow.constant(value=1.0,
                                        dtype=flow.float32,
                                        shape=(3, 3, 3))
            return flow.broadcast_like(x=x,
                                    like=like_tensor,
                                    broadcast_axes=(1, 2))


        x = np.random.randn(3, 1, 1).astype(np.float32)
        out = broadcast_like_Job(x)

        # out.shape (3, 3, 3)

    """
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
    x: oneflow_api.BlobDesc,
    mask: oneflow_api.BlobDesc,
    value: Union[float, int],
    name: Optional[str] = None,
) -> oneflow_api.BlobDesc:
    r"""Fill a blob with a given value according to the given mask.

    Args:
        x (oneflow_api.BlobDesc): Input Blob.
        mask (oneflow_api.BlobDesc): Composed with 0 and 1, the input blob 'x' will be
            filled with the given value where the mask is 1.
        value (Union[int, int]): The value to use for filling the input blob.
        name (Optional[str], optional): The name for the operation. Defaults to None.
    Attention:
        x and mask must be broadcastable to each other.
        mask must be int type (int8/int32/int64).

    Returns:
        oneflow_api.BlobDesc: The value-filled Blob

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


@oneflow_export("dim_gather")
def dim_gather(
    input: oneflow_api.BlobDesc,
    dim: int,
    index: oneflow_api.BlobDesc,
    name: Optional[str] = None,
) -> oneflow_api.BlobDesc:
    r""" This operator gathers elements from `input` according to `index` along with the axis `dim`.

    Take a 3-D blob as example, the output is specified by:

    .. code-block:: python

        output[i][j][k] = input[index[i][j][k]][j][k]  # if dim == 0
        output[i][j][k] = input[i][index[i][j][k]][k]  # if dim == 1
        output[i][j][k] = input[i][j][index[i][j][k]]  # if dim == 2


    The shape of `input` and `index` should be the same except in the `dim` dimension.

    That is, if `input` is a n-dimension blob with shape :math:`(x_0, x_1, \dots, x_{i-1}, x_i, x_{i+1}, \dots, x_n)`,
    and `dim = i`, then `index` must be a n-dimension blob with shape :math:`(x_0, x_1, \dots, x_{i-1}, k, x_{i+1}, \dots, x_n)`
    where :math:`k \geq 1`.

    The return Blob `output` will have the same shape with `index`.

    Args:
        input (oneflow_api.BlobDesc): The input blob
        dim (int): The axis along which to index
        index (oneflow_api.BlobDesc): The index blob of elements to gather
        name (Optional[str], optional): The name of the operation. Defaults to None.

    Returns:
        oneflow_api.BlobDesc: The elements gathered from `input` will be returned as the output Blob.

    For example:

    .. code-block:: python

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp

        @flow.global_function()
        def dim_gather_Job(input: tp.Numpy.Placeholder((2, 2), dtype=flow.float64),
                        index:tp.Numpy.Placeholder((2, 2), dtype=flow.int32))->tp.Numpy:
            return flow.dim_gather(input, 1, index)

        input = np.array([[1, 2], [3, 4]]).astype(np.float64)
        index = np.array([[1, 0], [0, 1]]).astype(np.int32)

        out = dim_gather_Job(input, index)
        # output
        # [[2. 1.]
        #  [3. 4.]]

    """
    if len(input.shape) != len(index.shape):
        raise ValueError("Dimensions of input and index should equal")

    for i in range(0, len(input.shape)):
        if dim == i:
            continue
        else:
            if input.shape[i] != index.shape[i]:
                raise ValueError(
                    "Dimensions of input and index should be same except at dim"
                )

    if dim >= len(index.shape):
        raise ValueError(
            "Value of dim is out of range(dim should be less than len(index.shape))"
        )

    return (
        flow.user_op_builder(
            name if name is not None else id_util.UniqueStr("DimGather_")
        )
        .Op("dim_gather")
        .Input("input", [input])
        .Input("index", [index])
        .Output("output")
        .Attr("dim", int(dim))
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


@oneflow_export("amp_white_identity")
def amp_white_identity(
    x: oneflow_api.BlobDesc, name: Optional[str] = None
) -> oneflow_api.BlobDesc:
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


@oneflow_export("zeros")
def zeros(
    shape: Sequence[int],
    dtype: Optional[dtype_util.dtype] = None,
    name: Optional[str] = None,
) -> oneflow_api.BlobDesc:
    """This operator creates a Tensor filled with the scalar value `0`.

    Args:
        shape (Sequence[int]): The shape of the Tensor.
        dtype (Optional[dtype_util.dtype], optional): The data type. Defaults to None.
        name (Optional[str], optional): The name for the operator. Defaults to None.

    Returns:
        oneflow_api.BlobDesc: The result Tensor filled with value `0`

    For example:

    .. code-block:: python

        import oneflow as flow
        import oneflow.typing as tp


        @flow.global_function()
        def zeros_job() -> tp.Numpy:
            return flow.zeros(shape=(2, 3), dtype=flow.float32)


        out = zeros_job()

        # output: [[0. 0. 0.]
        #          [0. 0. 0.]]

    """
    if name is None:
        name = id_util.UniqueStr("Zeros_")

    if dtype is None:
        dtype = flow.float32

    return flow.constant(value=0.0, shape=shape, dtype=dtype, name=name + "constant")


@oneflow_export("ones")
def ones(
    shape: Sequence[int],
    dtype: Optional[dtype_util.dtype] = None,
    name: Optional[str] = None,
) -> oneflow_api.BlobDesc:
    """This operator creates a Tensor filled with the scalar value `1`.

    Args:
        shape (Sequence[int]): The shape of the Tensor.
        dtype (Optional[dtype_util.dtype], optional): The data type. Defaults to None.
        name (Optional[str], optional): The name for the operator. Defaults to None.

    Returns:
        oneflow_api.BlobDesc: The result Blob filled with value `1`

    For example:

    .. code-block:: python

        import oneflow as flow
        import oneflow.typing as tp


        @flow.global_function()
        def ones_job() -> tp.Numpy:
            return flow.ones(shape=(2, 3), dtype=flow.float32)


        out = ones_job()

        # output: [[1. 1. 1.]
        #          [1. 1. 1.]]
    """
    if name is None:
        name = id_util.UniqueStr("Ones_")

    if dtype is None:
        dtype = flow.float32

    return flow.constant(value=1.0, shape=shape, dtype=dtype, name=name + "constant")
