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
import os
from typing import Optional, Sequence, Sized, Union

import oneflow as flow
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util
import oneflow.python.framework.id_util as id_util
import oneflow.python.framework.interpret_util as interpret_util
import oneflow.python.framework.remote_blob as remote_blob_util
from oneflow.python.oneflow_export import oneflow_export
import oneflow_api


def _gen_unique_name_if_need(name, default_name):
    if name is None:
        return id_util.UniqueStr(default_name)

    assert isinstance(name, str), name
    return name


def _check_axis(axis, shape):
    if axis is None:
        axis = list(range(len(shape)))

    if isinstance(axis, int):
        axis = [axis]

    assert isinstance(axis, (list, tuple)), "Invalid axis {}".format(axis)
    for x in axis:
        if x < 0:
            x += len(shape)
        assert x >= 0 and x < len(shape), "Invalid axis {}, len(shape): {}".format(
            axis, len(shape)
        )

    return axis


def _do_reduce(x, name, op_type_name, keepdims, axis):
    op = (
        flow.user_op_builder(name)
        .Op(op_type_name)
        .Input("input_tensor", [x])
        .Output("output_tensor")
        .Attr("axis", axis)
        .Attr("keepdims", keepdims)
        .Build()
    )
    return op.InferAndTryRun().SoleOutputBlob()


@oneflow_export("math.reduce_sum")
def reduce_sum(
    input_tensor: oneflow_api.BlobDesc,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
    name: Optional[str] = None,
) -> oneflow_api.BlobDesc:
    """This operator computes the sum of elements across dimensions of a tensor

    Args:
        input_tensor (oneflow_api.BlobDesc): A Blob
        axis (Optional[Union[int, Sequence[int]]], optional): The dimension along which the sum value is computed. Defaults to None.
        keepdims (bool, optional): Whether to keep the reduced dimension in the output Blob. Defaults to False.
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow_api.BlobDesc: The result of sum on the specified axis of input Blob

    For example: 

    .. code-block:: python 

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def reduce_sum_Job(x: tp.Numpy.Placeholder((3, 3))
        ) -> tp.Numpy:
            return flow.math.reduce_sum(x, axis=1, keepdims=True)


        x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).astype(np.float32)
        out = reduce_sum_Job(x)

        # out [[ 6.]
        #      [15.]
        #      [24.]]

    """
    name = _gen_unique_name_if_need(name, "ReduceSum_")

    axis = _check_axis(axis, input_tensor.shape)
    if len(axis) == 0:
        return input_tensor

    op = (
        flow.user_op_builder(name)
        .Op("reduce_sum")
        .Input("input_tensor", [input_tensor])
        .Output("output_tensor")
        .Attr("axis", axis)
        .Attr("keepdims", keepdims)
        .Build()
    )
    return op.InferAndTryRun().SoleOutputBlob()


@oneflow_export("math.reduce_any")
def reduce_any(
    x: oneflow_api.BlobDesc,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
    name: Optional[str] = None,
) -> oneflow_api.BlobDesc:
    """This operator computes the `logical or` of input Blob along the specified axis

    Args:
        x (oneflow_api.BlobDesc): A Blob
        axis (Optional[Union[int, Sequence[int]]], optional): The dimension along which the logical and value is computed. Defaults to None.
        keepdims (bool, optional): Whether to keep the reduced dimension in the output Blob. Defaults to False.
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow_api.BlobDesc: The result of logical or on the specified axis of input Blob

    Note: 

        The input Blob dtype is int8

    For example: 

    .. code-block:: python 

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def reduce_any_Job(x: tp.Numpy.Placeholder((3, 3), dtype=flow.int8)
        ) -> tp.Numpy:
            return flow.math.reduce_any(x, axis=1, keepdims=True)


        x = np.array([[1, 0, 0], [0, 0, 0], [1, 0, 1]]).astype(np.int8)
        out = reduce_any_Job(x)

        # out [[1]
        #      [0]
        #      [1]]

    """
    name = _gen_unique_name_if_need(name, "ReduceAny_")
    axis = _check_axis(axis, x.shape)
    if len(axis) == 0:
        return flow.math.not_equal(x, flow.constant_scalar(value=0.0, dtype=x.dtype))
    return _do_reduce(x, name, "reduce_any", keepdims, axis)


@oneflow_export("math.reduce_min")
def reduce_min(
    x: oneflow_api.BlobDesc,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
    name: Optional[str] = None,
) -> oneflow_api.BlobDesc:
    """This operator computes the minimum value of input Blob along the specified axis

    Args:
        x (oneflow_api.BlobDesc): A Blob
        axis (Optional[Union[int, Sequence[int]]], optional): The dimension along which the minimum value is computed. Defaults to None.
        keepdims (bool, optional): Whether to keep the reduced dimension in the output Blob. Defaults to False.
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow_api.BlobDesc: The result of minimum value on the specified axis of input Blob
    
    For example: 

    .. code-block:: python 

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def reduce_min_Job(x: tp.Numpy.Placeholder((3, 3))
        ) -> tp.Numpy:
            return flow.math.reduce_min(x, axis=1, keepdims=True)


        x = np.array([[2, 1, 3], [5, 3, 6], [7, 4, 9]]).astype(np.float32)
        out = reduce_min_Job(x)

        # out [[1.]
        #      [3.]
        #      [4.]]

    """
    name = _gen_unique_name_if_need(name, "ReduceMin_")
    axis = _check_axis(axis, x.shape)
    if len(axis) == 0:
        return x
    return _do_reduce(x, name, "reduce_min", keepdims, axis)


@oneflow_export("math.reduce_max")
def reduce_max(
    x: oneflow_api.BlobDesc,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
    name: Optional[str] = None,
) -> oneflow_api.BlobDesc:
    """This operator computes the maximum value of input Blob along the specified axis

    Args:
        x (oneflow_api.BlobDesc): A Blob
        axis (Optional[Union[int, Sequence[int]]], optional): The dimension along which the maximum value is computed. Defaults to None.
        keepdims (bool, optional): Whether to keep the reduced dimension in the output Blob. Defaults to False.
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow_api.BlobDesc: The result of maximum value on the specified axis of input Blob
    
    For example: 

    .. code-block:: python 

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def reduce_max_Job(x: tp.Numpy.Placeholder((3, 3))
        ) -> tp.Numpy:
            return flow.math.reduce_max(x, axis=1, keepdims=True)


        x = np.array([[2, 1, 4], [5, 3, 7], [7, 4, 9]]).astype(np.float32)
        out = reduce_max_Job(x)
        
        # out [[4.]
        #      [7.]
        #      [9.]]

    """
    name = _gen_unique_name_if_need(name, "ReduceMax_")
    axis = _check_axis(axis, x.shape)
    if len(axis) == 0:
        return x
    return _do_reduce(x, name, "reduce_max", keepdims, axis)


@oneflow_export("math.reduce_prod")
def reduce_prod(
    x: oneflow_api.BlobDesc,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
    name: Optional[str] = None,
) -> oneflow_api.BlobDesc:
    """This operator computes the product of input Blob along the specified axis

    Args:
        x (oneflow_api.BlobDesc): A Blob
        axis (Optional[Union[int, Sequence[int]]], optional): The dimension along which the product is computed. Defaults to None.
        keepdims (bool, optional): Whether to keep the reduced dimension in the output Blob. Defaults to False.
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow_api.BlobDesc: The result of product value on the specified axis of input Blob
    
    For example: 

    .. code-block:: python 

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def reduce_product_Job(x: tp.Numpy.Placeholder((3, 3))
        ) -> tp.Numpy:
            return flow.math.reduce_prod(x, axis=1, keepdims=True)


        x = np.array([[1, 2, 3], [3, 4, 5], [6, 3, 2]]).astype(np.float32)
        out = reduce_product_Job(x)

        # out [[ 6.]
        #      [60.]
        #      [36.]]

    """
    name = _gen_unique_name_if_need(name, "ReduceProd_")
    axis = _check_axis(axis, x.shape)
    if len(axis) == 0:
        return x
    return _do_reduce(x, name, "reduce_prod", keepdims, axis)


@oneflow_export("math.reduce_all")
def reduce_all(
    x: oneflow_api.BlobDesc,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
    name: Optional[str] = None,
) -> oneflow_api.BlobDesc:
    """This operator computes the `logical and` of input Blob along the specified axis

    Args:
        x (oneflow_api.BlobDesc): A Blob
        axis (Optional[Union[int, Sequence[int]]], optional): The dimension along which the logical and value is computed. Defaults to None.
        keepdims (bool, optional): Whether to keep the reduced dimension in the output Blob. Defaults to False.
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow_api.BlobDesc: The result of logical and value on the specified axis of input Blob
    
    Note: 

        The input Blob dtype is int8
    
    For example: 

    .. code-block:: python 

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def reduce_all_Job(x: tp.Numpy.Placeholder((3, 3), dtype=flow.int8)
        ) -> tp.Numpy:
            return flow.math.reduce_all(x, axis=1, keepdims=True)


        x = np.array([[1, 0, 0], [0, 0, 0], [1, 1, 1]]).astype(np.int8)
        out = reduce_all_Job(x)

        # out [[0]
        #      [0]
        #      [1]]

    """
    name = _gen_unique_name_if_need(name, "ReduceAll_")
    axis = _check_axis(axis, x.shape)
    if len(axis) == 0:
        return flow.math.not_equal(x, flow.constant_scalar(value=0.0, dtype=x.dtype))
    return _do_reduce(x, name, "reduce_all", keepdims, axis)


@oneflow_export("math.reduce_euclidean_norm")
def reduce_euclidean_norm(
    input_tensor: oneflow_api.BlobDesc,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
    name: Optional[str] = None,
) -> oneflow_api.BlobDesc:
    r"""This operator computes the Euclidean norm of input Blob along the specified axis

    The equation is: 

    .. math:: 

        out=\sqrt{\sum_{t=0}^{n} x_{t}^2}

    Args:
        input_tensor (oneflow_api.BlobDesc): A Blob
        axis (Optional[Union[int, Sequence[int]]], optional): The dimension along which the Euclidean norm is computed. Defaults to None.
        keepdims (bool, optional): Whether to keep the reduced dimension in the output Blob. Defaults to False.
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow_api.BlobDesc: The result of Euclidean norm on the specified axis of input Blob
    
    For example: 

    .. code-block:: python 

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def reduce_euclidean_norm_Job(x: tp.Numpy.Placeholder((3, 2))
        ) -> tp.Numpy:
            return flow.math.reduce_euclidean_norm(x, axis=1, keepdims=True)


        x = np.array([[3, 4], [5, 12], [8, 15]]).astype(np.float32)
        out = reduce_euclidean_norm_Job(x)

        # out [[ 5.]
        #      [13.]
        #      [17.]]

    """
    name = _gen_unique_name_if_need(name, "ReduceEuclideanNorm_")
    return flow.math.sqrt(
        flow.math.reduce_sum(
            flow.math.square(input_tensor, name + "_square"),
            axis,
            keepdims,
            name + "_reduce_sum",
        ),
        name + "_sqrt",
    )


@oneflow_export("math.reduce_logsumexp")
def reduce_logsumexp(
    input_tensor: oneflow_api.BlobDesc,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
    name: Optional[str] = None,
) -> oneflow_api.BlobDesc:
    r"""This operator computes the log of exponential sum of input Blob along the specified axis


    The equation is: 

    .. math:: 

        out = log(\sum_{t=0}^{t=n} e^{x_{t}})

    Args:
        input_tensor (oneflow_api.BlobDesc): A Blob
        axis (Optional[Union[int, Sequence[int]]], optional): The dimension along which the log of exponential sum is computed. Defaults to None.
        keepdims (bool, optional): Whether to keep the reduced dimension in the output Blob. Defaults to False.
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow_api.BlobDesc: The result of log of exponential sum on the specified axis of input Blob
    
    For example: 

    .. code-block:: python 

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def reduce_logsumexp_Job(x: tp.Numpy.Placeholder((3, 2))
        ) -> tp.Numpy:
            return flow.math.reduce_logsumexp(x, axis=1, keepdims=True)


        x = np.array([[0, 0], [1, 1], [2, 2]]).astype(np.float32)
        out = reduce_logsumexp_Job(x)

        # out [[0.6931472]
        #      [1.6931472]
        #      [2.6931472]]

    """
    name = _gen_unique_name_if_need(name, "ReduceLogSumExp_")
    axis = _check_axis(axis, input_tensor.shape)
    return flow.math.log(
        flow.math.reduce_sum(
            flow.math.exp(input_tensor, name + "_exp"),
            axis,
            keepdims,
            name + "_reduce_sum",
        ),
        name + "_log",
    )


@oneflow_export("math.reduce_std")
def reduce_std(
    input_tensor: oneflow_api.BlobDesc,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
    name: Optional[str] = None,
) -> oneflow_api.BlobDesc:
    r"""This operator computes the standard deviation of input Blob along the specified axis

    The equation is: 

    .. math:: 

        out=\sqrt{\frac{1}{n}*\sum_{i=1}^{n}(x_i-mean)^2}
    
    Args:
        input_tensor (oneflow_api.BlobDesc): A Blob
        axis (Optional[Union[int, Sequence[int]]], optional): The dimension along which the standard deviation is computed. Defaults to None.
        keepdims (bool, optional): Whether to keep the reduced dimension in the output Blob. Defaults to False.
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow_api.BlobDesc: The result of standard deviation on the specified axis of input Blob

    For example: 

    .. code-block:: python 
    
        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def reduce_std_Job(x: tp.Numpy.Placeholder((3, 3))
        ) -> tp.Numpy:
            return flow.math.reduce_std(x, axis=1, keepdims=True)


        x = np.array([[0, 5, 10], [5, 5, 5], [12, 3, 0]]).astype(np.float32)
        out = reduce_std_Job(x)

        # out [[4.0824833]
        #      [0.       ]
        #      [5.0990195]]

    """
    name = _gen_unique_name_if_need(name, "ReduceStd_")
    axis = _check_axis(axis, input_tensor.shape)
    if isinstance(axis, list) and len(axis) == 0:
        return flow.zeros_like(
            input_tensor, dtype=input_tensor.dtype, name=name + "_zeros_like"
        )
    return flow.math.sqrt(
        flow.math.reduce_variance(
            input_tensor, axis, keepdims, name + "_reduce_variance"
        ),
        name + "_sqrt",
    )


@oneflow_export("math.reduce_variance")
def reduce_variance(
    input_tensor: oneflow_api.BlobDesc,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
    name: Optional[str] = None,
) -> oneflow_api.BlobDesc:
    r"""This operator computes the variance of input Blob along the specified axis

    The equation is: 

    .. math:: 

        out=\frac{1}{n}*\sum_{i=1}^{n}(x_i-mean)^2

    Args:
        input_tensor (oneflow_api.BlobDesc): A Blob
        axis (Optional[Union[int, Sequence[int]]], optional): The dimension along which the variance is computed. Defaults to None.
        keepdims (bool, optional): Whether to keep the reduced dimension in the output Blob. Defaults to False.
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow_api.BlobDesc: The result of variance on the specified axis of input Blob
    
    For example: 

    .. code-block:: python 

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def reduce_variance_Job(x: tp.Numpy.Placeholder((3, 3))
        ) -> tp.Numpy:
            return flow.math.reduce_variance(x, axis=1, keepdims=True)


        x = np.array([[0, 5, 10], [5, 5, 5], [12, 3, 0]]).astype(np.float32)
        out = reduce_variance_Job(x)

        # out [[16.666668]
        #      [ 0.      ]
        #      [26.      ]]

    """
    name = _gen_unique_name_if_need(name, "ReduceVariance_")
    axis = _check_axis(axis, input_tensor.shape)
    if isinstance(axis, list) and len(axis) == 0:
        return flow.zeros_like(
            input_tensor, dtype=input_tensor.dtype, name=name + "_zeros_like"
        )
    return flow.math.subtract(
        flow.math.reduce_mean(
            flow.math.square(input_tensor, name + "_square_minuend"),
            axis,
            keepdims,
            name + "_reduce_mean_minuend",
        ),
        flow.math.square(
            flow.math.reduce_mean(
                input_tensor, axis, keepdims, name + "_reduce_mean_subtrahend"
            ),
            name + "_square_subtrahend",
        ),
        name + "_subtract",
    )
