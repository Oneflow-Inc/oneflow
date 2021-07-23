from functools import reduce
from typing import Iterable, List, Optional, Sequence, Union, Tuple
import numpy as np
import operator
import oneflow as flow
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util
import oneflow.framework.interpret_util as interpret_util
import oneflow.framework.id_util as id_util
import oneflow.framework.remote_blob as remote_blob_util
import oneflow._oneflow_internal


def infer_shape(x, shape):
    dim_index_need_infer = shape.index(-1) if shape.count(-1) == 1 else None
    in_elem_cnt = reduce(operator.mul, x.shape, 1)
    out_elem_cnt = reduce(operator.mul, shape, 1)
    if dim_index_need_infer is not None:
        assert in_elem_cnt % out_elem_cnt == 0
        shape[dim_index_need_infer] = int(abs(in_elem_cnt / out_elem_cnt))
    else:
        assert in_elem_cnt == out_elem_cnt
    return shape


def reshape_like(
    x: oneflow._oneflow_internal.BlobDesc,
    like: oneflow._oneflow_internal.BlobDesc,
    name: Optional[str] = None,
) -> oneflow._oneflow_internal.BlobDesc:
    """This operator reshapes the Blob x to be the same as Blob `like` .

    Args:
        x (oneflow._oneflow_internal.BlobDesc): The input Blob.
        like (oneflow._oneflow_internal.BlobDesc): A Blob.
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow._oneflow_internal.BlobDesc: The result Blob

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


def dynamic_reshape(
    x: oneflow._oneflow_internal.BlobDesc,
    shape: Sequence[int],
    name: Optional[str] = None,
) -> oneflow._oneflow_internal.BlobDesc:
    """This operator reshapes a dynamic blob.

    Args:
        x (oneflow._oneflow_internal.BlobDesc): The input Blob.
        shape (Sequence[int]): The output shape.
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow._oneflow_internal.BlobDesc: The result Blob.

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


def check_slice_tup_list(slice_tup_list, shape):
    ndim = len(shape)
    if not isinstance(slice_tup_list, (list, tuple)) or len(slice_tup_list) > ndim:
        raise ValueError(
            "slice_tup_list must be a list or tuple with length less than or equal to number of dimensions of input tensor"
        )
    if len(slice_tup_list) < ndim:
        slice_tup_list += type(slice_tup_list)(
            [(None, None, None)] * (ndim - len(slice_tup_list))
        )
    start_list = []
    stop_list = []
    step_list = []
    for (slice_tup, dim_size) in zip(slice_tup_list, shape):
        if not isinstance(slice_tup, (tuple, list)) or len(slice_tup) != 3:
            raise ValueError(
                "element of slice_tup_list must be a list or tuple with form (start, stop, step)"
            )
        if not all((isinstance(idx, int) or idx is None for idx in slice_tup)):
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
    return (start_list, stop_list, step_list)


def slice_v2(
    x: oneflow._oneflow_internal.BlobDesc,
    slice_tup_list: Sequence[Tuple[int, int, int]],
    name: Optional[str] = None,
) -> oneflow._oneflow_internal.BlobDesc:
    """Extracts a slice from a tensor.
    The `slice_tup_list` assigns the slice indices in each dimension, the format is (start, stop, step).
    The operator will slice the Blob according to the `slice_top_list`.

    Args:
        x: A `Blob`.
        slice_tup_list: A list of slice tuple, indicate each dimension slice (start, stop, step).
        name: A name for the operation (optional).

    Returns:
        oneflow._oneflow_internal.BlobDesc: The result Blob.

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
    (start, stop, step) = check_slice_tup_list(slice_tup_list, x.shape)
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


def GetSliceAttrs(slice_tup_list, input_shape):
    ndim = len(input_shape)
    if not (isinstance(slice_tup_list, (list, tuple)) and len(slice_tup_list) <= ndim):
        raise ValueError(
            "slice_tup_list must be a list or tuple with length less than or equal to number of dimensions of input tensor"
        )
    if len(slice_tup_list) < ndim:
        slice_tup_list += type(slice_tup_list)(
            [(None, None, None)] * (ndim - len(slice_tup_list))
        )
    start_list = []
    stop_list = []
    step_list = []
    for (slice_tup, dim_size) in zip(slice_tup_list, input_shape):
        if not (isinstance(slice_tup, (tuple, list)) and len(slice_tup) == 3):
            raise ValueError(
                "element of slice_tup_list must be a list or tuple with form (start, stop, step)"
            )
        if not all((isinstance(idx, int) or idx is None for idx in slice_tup)):
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
    return (start_list, stop_list, step_list)


def logical_slice(
    x: oneflow._oneflow_internal.BlobDesc,
    slice_tup_list: Sequence[Tuple[int, int, int]],
    name: Optional[str] = None,
) -> oneflow._oneflow_internal.BlobDesc:
    name = id_util.UniqueStr("LogicalSlice_") if name is None else name
    if not isinstance(name, str):
        raise ValueError("name must be a string")
    (start_list, stop_list, step_list) = GetSliceAttrs(slice_tup_list, x.shape)
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


def logical_slice_assign(
    x: oneflow._oneflow_internal.BlobDesc,
    value: oneflow._oneflow_internal.BlobDesc,
    slice_tup_list: Sequence[Tuple[int, int, int]],
    name: Optional[str] = None,
) -> oneflow._oneflow_internal.BlobDesc:
    name = id_util.UniqueStr("LogicalSliceAssign_") if name is None else name
    if not isinstance(name, str):
        raise ValueError("name must be a string")
    (start_list, stop_list, step_list) = GetSliceAttrs(slice_tup_list, x.shape)
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


def reverse(
    input: oneflow._oneflow_internal.BlobDesc,
    axis: Union[int, Sequence[int]],
    name: Optional[str] = None,
) -> oneflow._oneflow_internal.BlobDesc:
    """This operator reverses the elements on the assigned axis.

    Args:
        input (oneflow._oneflow_internal.BlobDesc): The input Blob.
        axis (Union[int, Sequence[int]]): The reverse axis.
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Raises:
        ValueError: The name must be a string.
        ValueError: The axis must be a int or a list/tuple of int.
        ValueError: The axis is out of range.

    Returns:
        oneflow._oneflow_internal.BlobDesc: The result Blob

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
    if not isinstance(axis, (tuple, list)) or not all(
        (isinstance(a, int) for a in axis)
    ):
        raise ValueError("axis must be a int or a list/tuple of int")
    ndim = len(input.shape)
    slice_tup_list = [(None, None, None)] * ndim
    for (i, a) in enumerate(axis):
        if a < 0:
            a += ndim
        if a < 0 or a >= ndim:
            raise ValueError("axis is out of range")
        slice_tup_list[a] = (None, None, -1)
    return slice_v2(input, slice_tup_list, name)


def concat(
    inputs: Optional[Sequence[oneflow._oneflow_internal.BlobDesc]] = None,
    axis: int = 0,
    max_dim_size: Optional[int] = None,
    name: Optional[str] = None,
    values: Optional[Sequence[oneflow._oneflow_internal.BlobDesc]] = None,
) -> oneflow._oneflow_internal.BlobDesc:
    """Concatenate two or more `Blob` s at specified axis.

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


def tensor_scatter_nd_update(
    params: oneflow._oneflow_internal.BlobDesc,
    indices: oneflow._oneflow_internal.BlobDesc,
    updates: oneflow._oneflow_internal.BlobDesc,
    name: Optional[str] = None,
) -> oneflow._oneflow_internal.BlobDesc:
    """This operator inserts the elements in `updates` according to the `indices` into the Blob `params`.

    Args:
        params (oneflow._oneflow_internal.BlobDesc): The input Blob.
        indices (oneflow._oneflow_internal.BlobDesc): The indice of `updates`. Its type should be `flow.int32`.
        updates (oneflow._oneflow_internal.BlobDesc): The update Blob.
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow._oneflow_internal.BlobDesc: The result Blob.

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


def tensor_scatter_nd_add(
    params: oneflow._oneflow_internal.BlobDesc,
    indices: oneflow._oneflow_internal.BlobDesc,
    updates: oneflow._oneflow_internal.BlobDesc,
    name: Optional[str] = None,
) -> oneflow._oneflow_internal.BlobDesc:
    """This operator adds elements from 'updates' to Blob 'params' based on the `indices`.

    Args:
        params (oneflow._oneflow_internal.BlobDesc): The input Blob.
        indices (oneflow._oneflow_internal.BlobDesc): The indice of `updates`. Its type should be `flow.int32`.
        updates (oneflow._oneflow_internal.BlobDesc): The update Blob.
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow._oneflow_internal.BlobDesc: The result Blob.

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


def elem_cnt(
    inputs: oneflow._oneflow_internal.BlobDesc,
    dtype: Optional[flow.dtype] = None,
    name: Optional[str] = None,
) -> oneflow._oneflow_internal.BlobDesc:
    """This operator returns the amount of elements in input Blob.

    Args:
        inputs (oneflow._oneflow_internal.BlobDesc): The input Blob.
        dtype (Optional[flow.dtype], optional): The data type. Defaults to None.
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow._oneflow_internal.BlobDesc: The result Blob. Its type is `ListNumpy`.

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
        op_conf.shape_elem_cnt_conf.data_type = oneflow._oneflow_internal.deprecated.GetProtoDtype4OfDtype(
            dtype
        )
    op_conf.shape_elem_cnt_conf.y = "y"
    interpret_util.Forward(op_conf)
    out_lbi = logical_blob_id_util.LogicalBlobId()
    setattr(out_lbi, "op_name", op_conf.name)
    setattr(out_lbi, "blob_name", "y")
    return remote_blob_util.RemoteBlob(out_lbi)


def sync_dynamic_resize(
    inputs: oneflow._oneflow_internal.BlobDesc,
    size: oneflow._oneflow_internal.BlobDesc,
    name: Optional[str] = None,
) -> oneflow._oneflow_internal.BlobDesc:
    """

    Args:
        inputs (oneflow._oneflow_internal.BlobDesc): The input Blob.
        size (oneflow._oneflow_internal.BlobDesc): The size of new Blob.
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow._oneflow_internal.BlobDesc: The result Blob. Its type is `ListNumpy`.

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


def generate_random_batch_permutation_indices(
    value: oneflow._oneflow_internal.BlobDesc,
    seed: Optional[int] = None,
    name: Optional[str] = None,
) -> oneflow._oneflow_internal.BlobDesc:
    """This operator generates a random permutation of indices in batch axis.

    Args:
        value (oneflow._oneflow_internal.BlobDesc): The input Blob.
        seed (Optional[int], optional): The random seed. Defaults to None.
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow._oneflow_internal.BlobDesc: The result Blob. Its type is `ListNumpy`.

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


def shuffle(
    value: oneflow._oneflow_internal.BlobDesc,
    seed: Optional[int] = None,
    name: Optional[str] = None,
) -> oneflow._oneflow_internal.BlobDesc:
    """This operator shuffle the elements in input Blob.

    Args:
        value (oneflow._oneflow_internal.BlobDesc): The input Blob.
        seed (Optional[int], optional): The random seed. Defaults to None.
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow._oneflow_internal.BlobDesc: The result Blob.

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


def identity(
    x: oneflow._oneflow_internal.BlobDesc, name: Optional[str] = None
) -> oneflow._oneflow_internal.BlobDesc:
    """This operator returns a `Blob` that has identical content and data type to input `Blob`.

    Analogous to `tf.identity <https://www.tensorflow.org/api_docs/python/tf/identity>`_

    Args:
        x (oneflow._oneflow_internal.BlobDesc): The input Blob.
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow._oneflow_internal.BlobDesc: The result Blob.

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


def identity_n(
    inputs: Sequence[oneflow._oneflow_internal.BlobDesc], name: Optional[str] = None
) -> List[oneflow._oneflow_internal.BlobDesc]:
    """This operator is similar to `oneflow.identity`. The difference is that the input and output
    of `identity_n` is `List`.

    Args:
        inputs (Iterable[oneflow._oneflow_internal.BlobDesc]): A List of input Blob.
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        List[oneflow._oneflow_internal.BlobDesc]: A list of result Blob.

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


def cast_to_static_shape(
    x: oneflow._oneflow_internal.BlobDesc, name: Optional[str] = None
) -> oneflow._oneflow_internal.BlobDesc:
    """This operator returns a `Blob` that has identical content and data type to input `Blob`, and whose shape is converted from dynamic to static

    Args:
        x (oneflow._oneflow_internal.BlobDesc): The input Blob which has dynamic shape.
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow._oneflow_internal.BlobDesc: The result Blob which is identical to input blob but has static shape.

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


def expand_dims(
    input: oneflow._oneflow_internal.BlobDesc, axis: int, name: Optional[str] = None
) -> oneflow._oneflow_internal.BlobDesc:
    """This operator inserts a dimention at the specified axis in the input Blob.
    The size of new dimension can only be 1, and the amount of element in return value is the same as Blob `input`.

    Args:
        input (oneflow._oneflow_internal.BlobDesc): The input Blob.
        axis (int): The specified dimension index.
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow._oneflow_internal.BlobDesc: The result Blob.

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


def dim_gather(
    input: oneflow._oneflow_internal.BlobDesc,
    dim: int,
    index: oneflow._oneflow_internal.BlobDesc,
    name: Optional[str] = None,
) -> oneflow._oneflow_internal.BlobDesc:
    """ This operator gathers elements from `input` according to `index` along with the axis `dim`.

    Take a 3-D blob as example, the output is specified by:

    .. code-block:: python

        output[i][j][k] = input[index[i][j][k]][j][k]  # if dim == 0
        output[i][j][k] = input[i][index[i][j][k]][k]  # if dim == 1
        output[i][j][k] = input[i][j][index[i][j][k]]  # if dim == 2


    The shape of `input` and `index` should be the same except in the `dim` dimension.

    That is, if `input` is a n-dimension blob with shape :math:`(x_0, x_1, \\dots, x_{i-1}, x_i, x_{i+1}, \\dots, x_n)`,
    and `dim = i`, then `index` must be a n-dimension blob with shape :math:`(x_0, x_1, \\dots, x_{i-1}, k, x_{i+1}, \\dots, x_n)`
    where :math:`k \\geq 1`.

    The return Blob `output` will have the same shape with `index`.

    Args:
        input (oneflow._oneflow_internal.BlobDesc): The input blob
        dim (int): The axis along which to index
        index (oneflow._oneflow_internal.BlobDesc): The index blob of elements to gather
        name (Optional[str], optional): The name of the operation. Defaults to None.

    Returns:
        oneflow._oneflow_internal.BlobDesc: The elements gathered from `input` will be returned as the output Blob.

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
        elif input.shape[i] != index.shape[i]:
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


def amp_white_identity(
    x: oneflow._oneflow_internal.BlobDesc, name: Optional[str] = None
) -> oneflow._oneflow_internal.BlobDesc:
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


def nvtx_start(
    x: oneflow._oneflow_internal.BlobDesc, mark_prefix: str, name: Optional[str] = None
) -> oneflow._oneflow_internal.BlobDesc:
    if name is None:
        name = id_util.UniqueStr("NvtxStart_")
    op = (
        flow.user_op_builder(name)
        .Op("nvtx_start")
        .Input("in", [x])
        .Output("out")
        .Attr("mark_prefix", str(mark_prefix))
        .Build()
    )
    return op.InferAndTryRun().SoleOutputBlob()


def nvtx_end(
    x: oneflow._oneflow_internal.BlobDesc, mark_prefix: str, name: Optional[str] = None
) -> oneflow._oneflow_internal.BlobDesc:
    if name is None:
        name = id_util.UniqueStr("NvtxEnd_")
    op = (
        flow.user_op_builder(name)
        .Op("nvtx_end")
        .Input("in", [x])
        .Output("out")
        .Attr("mark_prefix", str(mark_prefix))
        .Build()
    )
    return op.InferAndTryRun().SoleOutputBlob()
