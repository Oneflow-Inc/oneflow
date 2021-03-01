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

import os
from typing import Union, Optional, Sequence, List, Tuple

import oneflow as flow
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util
import oneflow.python.framework.interpret_util as interpret_util
import oneflow.python.framework.id_util as id_util
import oneflow.python.framework.remote_blob as remote_blob_util
import oneflow.python.framework.module as module_util
import oneflow.python.ops.math_unary_elementwise_ops as math_unary_elementwise_ops
from oneflow.python.oneflow_export import oneflow_export
from oneflow.python.ops.transpose_util import get_perm_when_transpose_axis_to_last_dim
from oneflow.python.ops.transpose_util import get_inversed_perm
import oneflow_api


@oneflow_export("math.add")
def add(
    x: Union[int, float, oneflow_api.BlobDesc],
    y: Union[int, float, oneflow_api.BlobDesc],
    name: Optional[str] = None,
) -> oneflow_api.BlobDesc:
    """Compute :math:`X + Y` element-wise, math.add supports broadcasting.
    The equation is:

    .. math::
        out = X + Y

    Args:
        x (Union[int, float, oneflow_api.BlobDesc]): A Blob.
        y (Union[int, float, oneflow_api.BlobDesc]): A Blob has the same type of x.
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow_api.BlobDesc: A Blob is added by x and y, and has the same type of x.

    For example:

    .. code-block:: python

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp

        @flow.global_function()
        def addJob(x: tp.Numpy.Placeholder((3, )),
                y: tp.Numpy.Placeholder((3, ))
        )->tp.Numpy:
            return flow.math.add(x, y)

        x = np.array([1, 2, 3]).astype(np.float32)
        y = np.array([1, 1, 1]).astype(np.float32)
        out = addJob(x, y)

        # out [2., 3., 4.]

    """
    if isinstance(x, (int, float)):
        return scalar_add(y, x, name)
    elif isinstance(y, (int, float)):
        return scalar_add(x, y, name)
    elif x.shape == y.shape and x.is_dynamic == y.is_dynamic:
        return element_wise_add(x, y, name)
    elif x.shape == (1,):
        return scalar_add_by_tensor(y, x, name)
    elif y.shape == (1,):
        return scalar_add_by_tensor(x, y, name)
    else:
        return broadcast_add(x, y, name)


def _recursive_build_add_n(inputs, name=None):
    inputs = list(inputs)
    kernel_max_inputs = 8
    if len(inputs) == 1:
        return inputs[0]
    elif len(inputs) <= kernel_max_inputs:
        return (
            flow.user_op_builder(
                name if name is not None else id_util.UniqueStr("AddN_")
            )
            .Op("add_n")
            .Input("in", inputs)
            .Output("out")
            .Build()
            .InferAndTryRun()
            .RemoteBlobList()[0]
        )
    else:
        assert len(inputs) > kernel_max_inputs
        new_inputs = inputs[kernel_max_inputs:]
        new_inputs.append(_recursive_build_add_n(inputs[:kernel_max_inputs]))
        return _recursive_build_add_n(new_inputs)


@oneflow_export("math.add_n")
def add_n(
    inputs: Sequence[oneflow_api.BlobDesc], name: Optional[str] = None
) -> oneflow_api.BlobDesc:
    """Add all the input tensors in element-wise.

    Args:
        inputs (Sequence[oneflow_api.BlobDesc]): A list of Blob, each Blob has the same shape and type.
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow_api.BlobDesc: The sum of the inputs, has the same shape and type as the elements of inputs.

    For example:

    .. code-block:: python

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp

        @flow.global_function()
        def add_n_Job(x: tp.Numpy.Placeholder((3, )),
                    y: tp.Numpy.Placeholder((3, ))
        )->tp.Numpy:
            return flow.math.add_n([x, y])

        x = np.array([1, 2, 3]).astype(np.float32)
        y = np.array([1, 1, 1]).astype(np.float32)
        out = add_n_Job(x, y)
        print(out)

        # out [2., 3., 4.]

    """
    return _recursive_build_add_n(inputs, name)


@oneflow_export("math.subtract")
def subtract(
    x: Union[int, float, oneflow_api.BlobDesc],
    y: Union[int, float, oneflow_api.BlobDesc],
    name: Optional[str] = None,
) -> oneflow_api.BlobDesc:
    """Compute :math:`X - Y` element-wise.

    The equation is:

    .. math::
        out = X - Y

    Args:
        x (Union[int, float, oneflow_api.BlobDesc]): A Blob.
        y (Union[int, float, oneflow_api.BlobDesc]): A Blob has the same type of x.
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow_api.BlobDesc: A Blob after subtracting, has the same type as x.

    For example:

    .. code-block:: python

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp

        @flow.global_function()
        def subtractJob(x: tp.Numpy.Placeholder((3, )),
                        y: tp.Numpy.Placeholder((3, ))
        )->tp.Numpy:
            return flow.math.subtract(x, y)

        x = np.array([1, 2, 3]).astype(np.float32)
        y = np.array([2, 4, 1]).astype(np.float32)
        out = subtractJob(x, y)

        # out [-1., -2., 2.]

    """
    if isinstance(x, (int, float)):
        return scalar_add(-1 * y, x, name)
    elif isinstance(y, (int, float)):
        return scalar_add(x, -1 * y, name)
    elif x.shape == y.shape:
        # TODO: add element-wise op
        return broadcast_sub(x, y, name)
    elif x.shape == (1,):
        return scalar_sub_by_tensor(y, x, name)
    elif y.shape == (1,):
        return scalar_sub_by_tensor(x, y, name)
    else:
        return broadcast_sub(x, y, name)


@oneflow_export("math.multiply")
def multiply(
    x: Union[int, float, oneflow_api.BlobDesc],
    y: Union[int, float, oneflow_api.BlobDesc],
    name: Optional[str] = None,
) -> oneflow_api.BlobDesc:
    r"""Compute :math:`x \times y` element-wise.

    The equation is:

    .. math::
        out = X \times Y

    Args:
        x (Union[int, float, oneflow_api.BlobDesc]): A Blob.
        y (Union[int, float, oneflow_api.BlobDesc]): A Blob has the same type of x.
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow_api.BlobDesc: A Blob after multiplying, has the same type as x.

    For example:

    .. code-block:: python

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp

        @flow.global_function()
        def multiplyJob(x: tp.Numpy.Placeholder((3, )),
                        y: tp.Numpy.Placeholder((3, ))
        )->tp.Numpy:
            return flow.math.multiply(x, y)

        x = np.array([1, 2, 3]).astype(np.float32)
        y = np.array([2, 3, 3]).astype(np.float32)
        out = multiplyJob(x, y)

        # out [2., 6., 9.]

    """
    if isinstance(x, (int, float)):
        return scalar_mul(y, x, name)
    elif isinstance(y, (int, float)):
        return scalar_mul(x, y, name)
    elif x.shape == y.shape:
        return element_wise_mul(x, y, name)
    elif x.shape == (1,):
        return scalar_mul_by_tensor(y, x, name)
    elif y.shape == (1,):
        return scalar_mul_by_tensor(x, y, name)
    else:
        return broadcast_mul(x, y, name)


@oneflow_export("math.divide")
def divide(
    x: Union[int, float, oneflow_api.BlobDesc],
    y: Union[int, float, oneflow_api.BlobDesc],
    name: Optional[str] = None,
) -> oneflow_api.BlobDesc:
    r"""Computes the division of x by y.

    The equation is:

    .. math::
        out = \frac{X}{Y}

    Args:
        x (Union[int, float, oneflow_api.BlobDesc]): A Blob.
        y (Union[int, float, oneflow_api.BlobDesc]): A Blob.
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow_api.BlobDesc: A Blob with same shape as input x.

    For example:

    .. code-block:: python

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp

        @flow.global_function()
        def divideJob(x: tp.Numpy.Placeholder((3, )),
                    y: tp.Numpy.Placeholder((3, ))
        )->tp.Numpy:
            return flow.math.divide(x, y)

        x = np.array([25, 16, 9]).astype(np.float32)
        y = np.array([10, 4, 2]).astype(np.float32)
        out = divideJob(x, y)

        # out [2.5, 4., 4.5]

    """
    if isinstance(x, (int, float)):
        return scalar_mul(math_unary_elementwise_ops.reciprocal_no_nan(y), x, name)
    elif isinstance(y, (int, float)):
        if y == 0 or y == 0.0:
            y = 0.0
        else:
            y = 1.0 / (float(y))
        return scalar_mul(x, y, name)
    elif x.shape == y.shape:
        # TODO: add element-wise op
        return broadcast_div(x, y, name)
    elif x.shape == (1,):
        return scalar_div_by_tensor(y, x, name)
    elif y.shape == (1,):
        return scalar_div_by_tensor(x, y, name)
    else:
        return broadcast_div(x, y, name)


@oneflow_export("math.mod")
def floor_mod(
    x: Union[int, float, oneflow_api.BlobDesc],
    y: Union[int, float, oneflow_api.BlobDesc],
    name: Optional[str] = None,
) -> oneflow_api.BlobDesc:
    r"""This operator mods two Blobs.

    The equation is:

    .. math::
        out = X \bmod Y

    Args:
        x (Union[int, float, oneflow_api.BlobDesc]): A Blob
        y (Union[int, float, oneflow_api.BlobDesc]): A Blob has the same type of x
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Raises:
        NotImplementedError: x must be an int or a float
        NotImplementedError: y must be an int or a float

    Returns:
        oneflow_api.BlobDesc: A Blob with same type as input x.

    For example:

    .. code-block:: python

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp

        @flow.global_function()
        def modJob(x: tp.Numpy.Placeholder((3, )),
                y: tp.Numpy.Placeholder((3, ))
        )->tp.Numpy:
            return flow.math.mod(x, y)

        x = np.array([16, 9, 5]).astype(np.float32)
        y = np.array([6, 4, 3]).astype(np.float32)
        out = modJob(x, y)

        # out [4., 1., 2.]

    """
    if isinstance(x, (int, float)):
        raise NotImplementedError
    elif isinstance(y, (int, float)):
        raise NotImplementedError
    elif x.shape == y.shape:
        # TODO: add element-wise op
        return broadcast_floor_mod(x, y, name)
    else:
        return broadcast_floor_mod(x, y, name)


def scalar_add(x, operand, name=None):
    if name is None:
        name = id_util.UniqueStr("ScalarAdd_")
    builder = flow.user_op_builder(name).Op("scalar_add").Input("in", [x]).Output("out")
    if isinstance(operand, int):
        builder = (
            builder.Attr("has_int_operand", True)
            .Attr("has_float_operand", False)
            .Attr("int_operand", operand)
            .Attr("float_operand", 0.0)
        )
    elif isinstance(operand, float):
        builder = (
            builder.Attr("has_int_operand", False)
            .Attr("has_float_operand", True)
            .Attr("int_operand", 0)
            .Attr("float_operand", operand)
        )
    return builder.Build().InferAndTryRun().RemoteBlobList()[0]


def scalar_add_by_tensor(x, scalar, name=None):
    return (
        flow.user_op_builder(name or id_util.UniqueStr("ScalarAddByTensor_"))
        .Op("scalar_add_by_tensor")
        .Input("x", [x])
        .Input("scalar", [scalar])
        .Output("y")
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


def element_wise_add(x, y, name=None):
    return flow.math.add_n([x, y], name)


def build_broadcast_binary_op(math_op, x, y, name=None):
    if name is None:
        name = id_util.UniqueStr(math_op + "_")
    return (
        flow.user_op_builder(name)
        .Op(math_op)
        .Input("x", [x])
        .Input("y", [y])
        .Output("z")
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


def broadcast_add(x, y, name=None):
    return build_broadcast_binary_op("broadcast_add", x, y, name)


def broadcast_sub(x, y, name=None):
    return build_broadcast_binary_op("broadcast_sub", x, y, name)


def scalar_sub_by_tensor(x, scalar, name=None):
    return (
        flow.user_op_builder(name or id_util.UniqueStr("ScalarSubByTensor_"))
        .Op("scalar_sub_by_tensor")
        .Input("x", [x])
        .Input("scalar", [scalar])
        .Output("y")
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


def element_wise_mul(x, y, name=None):
    return (
        flow.user_op_builder(name or id_util.UniqueStr("ElementWiseMul_"))
        .Op("multiply")
        .Input("x", [x])
        .Input("y", [y])
        .Output("out")
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


def broadcast_mul(x, y, name=None):
    return build_broadcast_binary_op("broadcast_mul", x, y, name)


def scalar_mul(x, operand, name=None):
    if name is None:
        name = id_util.UniqueStr("ScalarMul_")
    builder = flow.user_op_builder(name).Op("scalar_mul").Input("in", [x]).Output("out")
    if isinstance(operand, int):
        builder = (
            builder.Attr("has_int_operand", True)
            .Attr("has_float_operand", False)
            .Attr("int_operand", operand)
            .Attr("float_operand", 0.0)
        )
    elif isinstance(operand, float):
        builder = (
            builder.Attr("has_int_operand", False)
            .Attr("has_float_operand", True)
            .Attr("int_operand", 0)
            .Attr("float_operand", operand)
        )
    return builder.Build().InferAndTryRun().RemoteBlobList()[0]


def scalar_mul_by_tensor(x, scalar, name=None):
    return (
        flow.user_op_builder(name or id_util.UniqueStr("ScalarMulByTensor_"))
        .Op("scalar_mul_by_tensor")
        .Input("x", [x])
        .Input("scalar", [scalar])
        .Output("y")
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


def broadcast_div(x, y, name=None):
    return build_broadcast_binary_op("broadcast_div", x, y, name)


def scalar_div_by_tensor(x, scalar, name=None):
    return (
        flow.user_op_builder(name or id_util.UniqueStr("ScalarDivByTensor_"))
        .Op("scalar_div_by_tensor")
        .Input("x", [x])
        .Input("scalar", [scalar])
        .Output("y")
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


def broadcast_floor_mod(x, y, name=None):
    return build_broadcast_binary_op("broadcast_floor_mod", x, y, name)


@oneflow_export("math.gelu")
def gelu(x: oneflow_api.BlobDesc, name: Optional[str] = None) -> oneflow_api.BlobDesc:
    r"""Gelu activation operator.

    The equation is:

    .. math::
        out = 0.5 * x * (1 + tanh(\sqrt{\frac{2}{\pi}} * (x + 0.044715x^{3})))

    Args:
        x (oneflow_api.BlobDesc): Input Blob
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow_api.BlobDesc: A Blob.

    For example:

    .. code-block:: python

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp

        @flow.global_function()
        def geluJob(x: tp.Numpy.Placeholder((3, ))
        )->tp.Numpy:
            return flow.math.gelu(x)

        x = np.array([-0.5, 0, 0.5]).astype(np.float32)
        out = geluJob(x)

        # out [-0.15426877, 0., 0.34573123]

    """
    return (
        flow.user_op_builder(name if name is not None else id_util.UniqueStr("Gelu_"))
        .Op("gelu")
        .Input("in", [x])
        .Output("out")
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


@oneflow_export("math.relu", "nn.relu")
def relu(x: oneflow_api.BlobDesc, name: Optional[str] = None) -> oneflow_api.BlobDesc:
    r"""Relu activation

    The equation is:

    .. math::
        out = max(X, 0)

    Args:
        x (oneflow_api.BlobDesc): Input Blob
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow_api.BlobDesc: An activated Blob.

    For example:

    .. code-block:: python

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp

        @flow.global_function()
        def reluJob(x: tp.Numpy.Placeholder((3, ))
        )->tp.Numpy:
            return flow.math.relu(x)

        x = np.array([-1, 0, 5]).astype(np.float32)
        out = reluJob(x)

        # out [0., 0., 5.]

    """

    return (
        flow.user_op_builder(name if name is not None else id_util.UniqueStr("Relu_"))
        .Op("relu")
        .Input("in", [x])
        .Output("out")
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


@oneflow_export("math.sigmoid")
def sigmoid(
    x: oneflow_api.BlobDesc, name: Optional[str] = None
) -> oneflow_api.BlobDesc:
    r"""Sigmoid activation

    The equation is:

    .. math::
        out = \frac{1}{1 + e^{-x}}

    Args:
        x (oneflow_api.BlobDesc): Input Blob
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow_api.BlobDesc: An activated Blob.

    For example:

    .. code-block:: python

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp

        @flow.global_function()
        def sigmoidJob(x: tp.Numpy.Placeholder((3, ))
        )->tp.Numpy:
            return flow.math.sigmoid(x)

        x = np.array([-1, 0, 1]).astype(np.float32)
        out = sigmoidJob(x)

        # out [0.26894143, 0.5, 0.7310586]

    """
    return (
        flow.user_op_builder(
            name if name is not None else id_util.UniqueStr("Sigmoid_")
        )
        .Op("sigmoid")
        .Input("in", [x])
        .Output("out")
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


@oneflow_export("math.sigmoid_grad")
def sigmoid_grad(
    y: oneflow_api.BlobDesc, dy: oneflow_api.BlobDesc, name: Optional[str] = None,
) -> oneflow_api.BlobDesc:
    return (
        flow.user_op_builder(
            name if name is not None else id_util.UniqueStr("SigmoidGrad_")
        )
        .Op("sigmoid_grad")
        .Input("y", [y])
        .Input("dy", [dy])
        .Output("dx")
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


@oneflow_export("math.unsorted_segment_sum", "unsorted_segment_sum")
def unsorted_segment_sum(
    data: oneflow_api.BlobDesc,
    segment_ids: oneflow_api.BlobDesc,
    num_segments: int,
    axis: int = 0,
    name: Optional[str] = None,
) -> oneflow_api.BlobDesc:
    r"""Computes the sum along segments of a Blob.

    Args:
        data (oneflow_api.BlobDesc): Input Blob
        segment_ids (oneflow_api.BlobDesc): A Blob should be the size of the first dimension, with consecutive IDs in the range 0 to k (k < d0).
        num_segments (int): num_segments should equal the number of distinct segment IDs.
        axis (int, optional): The axis of data. Defaults to 0.
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow_api.BlobDesc: A Blob with the same type of data.

    For example:

    .. code-block:: python

        # Example 1:
        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp

        @flow.global_function()
        def unsorted_segment_sumJob(data: tp.Numpy.Placeholder((3, 4)),
                                    segment_ids: tp.Numpy.Placeholder((4, ), dtype=flow.int32)
        )->tp.Numpy:
            return flow.math.unsorted_segment_sum(data, segment_ids, num_segments=2, axis=1)

        input_blob = np.array([[1, 2, 3, 4],
                               [5, 6, 7 ,8],
                               [9, 10, 11, 12]]).astype(np.float32)
        segment_ids = np.array([0, 1, 0, 1]).astype(np.int32)
        out = unsorted_segment_sumJob(input_blob, segment_ids)

        # out [[ 4.  6.]
        #      [12. 14.]
        #      [20. 22.]]

        # Example 2
        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp

        @flow.global_function()
        def unsorted_segment_sumJob(data: tp.Numpy.Placeholder((3, 4)),
                                    segment_ids: tp.Numpy.Placeholder((3, ), dtype=flow.int32)
        )->tp.Numpy:
            return flow.math.unsorted_segment_sum(data, segment_ids, num_segments=2, axis=0)

        input_blob = np.array([[1, 2, 3, 4],
                               [5, 6, 7 ,8],
                               [9, 10, 11, 12]]).astype(np.float32)
        segment_ids = np.array([0, 1, 0]).astype(np.int32)
        out = unsorted_segment_sumJob(input_blob, segment_ids)

        #  out [[10. 12. 14. 16.]
        #       [ 5.  6.  7.  8.]]

    """
    return (
        flow.user_op_builder(
            name if name is not None else id_util.UniqueStr("UnsortedSegmentSum_")
        )
        .Op("unsorted_segment_sum")
        .Input("data", [data])
        .Input("segment_ids", [segment_ids])
        .Output("out")
        .Attr("axis", int(axis))
        .Attr("num_segments", int(num_segments))
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


@oneflow_export("math.unsorted_segment_sum_like", "unsorted_segment_sum_like")
def unsorted_segment_sum_like(
    data: oneflow_api.BlobDesc,
    segment_ids: oneflow_api.BlobDesc,
    like: oneflow_api.BlobDesc,
    axis: int = 0,
    name: Optional[str] = None,
) -> oneflow_api.BlobDesc:
    r"""Computes the sum along segments of a Blob, the output shape is the same as the `like` Blob.

    Args:
        data (oneflow_api.BlobDesc): Input Blob
        segment_ids (oneflow_api.BlobDesc): A Blob should be the size of the first dimension, with consecutive IDs in the range 0 to k (k < d0).
        like (oneflow_api.BlobDesc): The input Blob which specifies shape
        axis (int, optional): The axis of data. Defaults to 0.
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow_api.BlobDesc: A Blob.

    For example:

    .. code-block:: python

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp

        @flow.global_function()
        def unsorted_segment_sum_like_Job(data: tp.Numpy.Placeholder((3, 4)),
                                        segment_ids: tp.Numpy.Placeholder((3, ), dtype=flow.int32),
                                        like: tp.Numpy.Placeholder((2, 4), dtype=flow.float32)
        )->tp.Numpy:
            return flow.math.unsorted_segment_sum_like(data, segment_ids, like, axis=0)

        input_blob = np.array([[1, 2, 3, 4],
                            [5, 6, 7 ,8],
                            [9, 10, 11, 12]]).astype(np.float32)
        segment_ids = np.array([0, 1, 0]).astype(np.int32)
        like = np.zeros(shape=(2, 4), dtype=np.float32)

        out = unsorted_segment_sum_like_Job(input_blob, segment_ids, like)

        # out [[10. 12. 14. 16.]
        #      [ 5.  6.  7.  8.]]

    """
    return (
        flow.user_op_builder(
            name if name is not None else id_util.UniqueStr("UnsortedSegmentSumLike_")
        )
        .Op("unsorted_segment_sum_like")
        .Input("data", [data])
        .Input("segment_ids", [segment_ids])
        .Input("like", [like])
        .Output("out")
        .Attr("axis", int(axis))
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


@oneflow_export("math.unsorted_batch_segment_sum", "unsorted_batch_segment_sum")
def unsorted_batch_segment_sum(
    data: oneflow_api.BlobDesc,
    segment_ids: oneflow_api.BlobDesc,
    num_segments: int,
    name: Optional[str] = None,
) -> oneflow_api.BlobDesc:
    r"""It is similar with `unsorted_segment_sum`, the difference is that `unsorted_batch_segment_sum` brings a `batch axis`. We can do the segment sum in different batch of data.

    For example, the segment id is like:

    .. code-block:: python

        [[0 0 0 1 2 2 3 3],
         [0 0 1 1 2 3 3 3]]

    Args:
        data (oneflow_api.BlobDesc): Input Blob
        segment_ids (oneflow_api.BlobDesc): A Blob with shape (d0, d1). The d0, d1 are the first and second dimension of data.
        num_segments (int): num_segments should equal the number of distinct segment IDs.
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow_api.BlobDesc: A Blob.

    For example:

    .. code-block:: python

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp

        @flow.global_function()
        def unsorted_batch_segment_sum_Job(data: tp.Numpy.Placeholder((3, 4)),
                                        segment_ids: tp.Numpy.Placeholder((3, 4), dtype=flow.int32)
        )->tp.Numpy:
            return flow.math.unsorted_batch_segment_sum(data, segment_ids, 2)

        input_blob = np.array([[1, 2, 3, 4],
                            [1, 2, 3 ,4],
                            [1, 2, 3, 4]]).astype(np.float32)
        segment_ids = np.array([[0, 0, 0, 1],
                                [0, 0, 1, 0],
                                [0, 1, 0, 0]]).astype(np.int32)
        out = unsorted_batch_segment_sum_Job(input_blob, segment_ids)

        # out [[6. 4.]
        #      [7. 3.]
        #      [8. 2.]]

    """
    return (
        flow.user_op_builder(
            name if name is not None else id_util.UniqueStr("UnsortedBatchSegmentSum_")
        )
        .Op("unsorted_batch_segment_sum")
        .Input("data", [data])
        .Input("segment_ids", [segment_ids])
        .Output("out")
        .Attr("num_segments", int(num_segments))
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


@oneflow_export("cast")
def cast(
    x: oneflow_api.BlobDesc, dtype: flow.dtype, name: Optional[str] = None
) -> oneflow_api.BlobDesc:
    r"""The op takes input x and casts it to the output with `dtype`

    Args:
        x (oneflow_api.BlobDesc): Input Blob
        dtype (flow.dtype): Data type of the output
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow_api.BlobDesc: A Blob

    For example:

    .. code-block:: python

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp

        @flow.global_function()
        def cast_Job(x: tp.Numpy.Placeholder((3, ), dtype=flow.float32)
        )->tp.Numpy:
            return flow.cast(x, dtype=flow.int32)

        x = np.array([1, 2, 3]).astype(np.float32)
        out = cast_Job(x)

        # out.dtype = "int32"

    """
    if x.dtype == dtype:
        return x
    if name is None:
        name = id_util.UniqueStr("Cast_")

    return (
        flow.user_op_builder(name)
        .Op("cast")
        .Input("in", [x])
        .Output("out")
        .Attr("dtype", dtype)
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


@oneflow_export("math.equal")
def equal(
    x: oneflow_api.BlobDesc, y: oneflow_api.BlobDesc, name: Optional[str] = None
) -> oneflow_api.BlobDesc:
    r"""Returns the truth value of :math:`{x}=={y}` element-wise.

    Args:
        x (oneflow_api.BlobDesc): A Blob
        y (oneflow_api.BlobDesc): A Blob
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow_api.BlobDesc: A Blob with int8 type.

    For example:

    .. code-block:: python

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp

        @flow.global_function()
        def equal_Job(x: tp.Numpy.Placeholder((3, )),
                    y: tp.Numpy.Placeholder((3, ))
        )->tp.Numpy:
            return flow.math.equal(x, y)

        x = np.array([1, 2, 3]).astype(np.float32)
        y = np.array([1, 2, 1]).astype(np.float32)
        out = equal_Job(x, y)

        # out [1 1 0]

    """
    return build_broadcast_binary_op("broadcast_equal", x, y, name)


@oneflow_export("math.not_equal")
def not_equal(
    x: oneflow_api.BlobDesc, y: oneflow_api.BlobDesc, name: Optional[str] = None
) -> oneflow_api.BlobDesc:
    r"""Returns the truth value of :math:`{x}!={y}` element-wise.

    Args:
        x (oneflow_api.BlobDesc): A Blob
        y (oneflow_api.BlobDesc): A Blob
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow_api.BlobDesc: A Blob with int8 type.

    For example:

    .. code-block:: python

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp

        @flow.global_function()
        def not_equal_Job(x: tp.Numpy.Placeholder((3, )),
                        y: tp.Numpy.Placeholder((3, ))
        )->tp.Numpy:
            return flow.math.not_equal(x, y)

        x = np.array([1, 2, 3]).astype(np.float32)
        y = np.array([1, 2, 1]).astype(np.float32)
        out = not_equal_Job(x, y)

        # out [0 0 1]

    """
    return build_broadcast_binary_op("broadcast_not_equal", x, y, name)


@oneflow_export("math.less")
def less(
    x: oneflow_api.BlobDesc, y: oneflow_api.BlobDesc, name: Optional[str] = None
) -> oneflow_api.BlobDesc:
    r"""Returns the truth value of :math:`x < y` element-wise.

    Args:
        x (oneflow_api.BlobDesc): A Blob
        y (oneflow_api.BlobDesc): A Blob
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow_api.BlobDesc: A Blob with int8 type.

    For example:

    .. code-block:: python

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp

        @flow.global_function()
        def less_Job(x: tp.Numpy.Placeholder((3, )),
                    y: tp.Numpy.Placeholder((3, ))
        )->tp.Numpy:
            return flow.math.less(x, y)

        x = np.array([1, 2, 3]).astype(np.float32)
        y = np.array([1, 2, 4]).astype(np.float32)
        out = less_Job(x, y)

        # out [0 0 1]

    """
    return build_broadcast_binary_op("broadcast_less", x, y, name)


@oneflow_export("math.less_equal")
def less_equal(
    x: oneflow_api.BlobDesc, y: oneflow_api.BlobDesc, name: Optional[str] = None
) -> oneflow_api.BlobDesc:
    r"""Returns the truth value of :math:`x <= y` element-wise.

    Args:
        x (oneflow_api.BlobDesc): A Blob
        y (oneflow_api.BlobDesc): A Blob
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow_api.BlobDesc: A Blob with int8 type.

    For example:

    .. code-block:: python

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp

        @flow.global_function()
        def less_equal_Job(x: tp.Numpy.Placeholder((3, )),
                        y: tp.Numpy.Placeholder((3, ))
        )->tp.Numpy:
            return flow.math.less_equal(x, y)

        x = np.array([1, 2, 3]).astype(np.float32)
        y = np.array([1, 1, 4]).astype(np.float32)
        out = less_equal_Job(x, y)

        # out [1 0 1]

    """
    return build_broadcast_binary_op("broadcast_less_equal", x, y, name)


@oneflow_export("math.greater")
def greater(
    x: oneflow_api.BlobDesc, y: oneflow_api.BlobDesc, name: Optional[str] = None
) -> oneflow_api.BlobDesc:
    r"""Returns the truth value of :math:`x > y` element-wise.

    Args:
        x (oneflow_api.BlobDesc): A Blob
        y (oneflow_api.BlobDesc): A Blob
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow_api.BlobDesc: A Blob with int8 type.

    For example:

    .. code-block:: python

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp

        @flow.global_function()
        def greater_Job(x: tp.Numpy.Placeholder((3, )),
                        y: tp.Numpy.Placeholder((3, ))
        )->tp.Numpy:
            return flow.math.greater(x, y)

        x = np.array([1, 1, 4]).astype(np.float32)
        y = np.array([1, 2, 3]).astype(np.float32)
        out = greater_Job(x, y)

        # out [0 0 1]

    """
    return build_broadcast_binary_op("broadcast_greater", x, y, name)


@oneflow_export("math.greater_equal")
def greater_equal(
    x: oneflow_api.BlobDesc, y: oneflow_api.BlobDesc, name: Optional[str] = None
) -> oneflow_api.BlobDesc:
    r"""Returns the truth value of :math:`x >= y` element-wise.

    Args:
        x (oneflow_api.BlobDesc): A Blob
        y (oneflow_api.BlobDesc): A Blob
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow_api.BlobDesc: A Blob with int8 type.

    For example:

    .. code-block:: python

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp

        @flow.global_function()
        def greater_equal_Job(x: tp.Numpy.Placeholder((3, )),
                            y: tp.Numpy.Placeholder((3, ))
        )->tp.Numpy:
            return flow.math.greater_equal(x, y)

        x = np.array([1, 1, 4]).astype(np.float32)
        y = np.array([1, 2, 3]).astype(np.float32)
        out = greater_equal_Job(x, y)

        # out [1 0 1]

    """
    return build_broadcast_binary_op("broadcast_greater_equal", x, y, name)


@oneflow_export("math.logical_and")
def logical_and(
    x: oneflow_api.BlobDesc, y: oneflow_api.BlobDesc, name: Optional[str] = None
) -> oneflow_api.BlobDesc:
    r"""Logical AND function.

    Each element is calculated by:

    .. math::
        out = X \land Y

    Args:
        x (oneflow_api.BlobDesc): A Blob
        y (oneflow_api.BlobDesc): A Blob
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow_api.BlobDesc: A Blob with int8 type.

    For example:

    .. code-block:: python

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp

        @flow.global_function()
        def logical_and_Job(x: tp.Numpy.Placeholder((3, )),
                            y: tp.Numpy.Placeholder((3, ))
        )->tp.Numpy:
            return flow.math.logical_and(x, y)

        x = np.array([1, 0, 1]).astype(np.float32)
        y = np.array([0, 0, 1]).astype(np.float32)
        out = logical_and_Job(x, y)

        # out [0 0 1]

    """
    return build_broadcast_binary_op("broadcast_logical_and", x, y, name)


@oneflow_export("math.minimum")
def minimum(
    x: oneflow_api.BlobDesc, y: oneflow_api.BlobDesc, name: Optional[str] = None
) -> oneflow_api.BlobDesc:
    r"""Returns the min of x and y element-wise, this op supports broadcasting.

    Args:
        x (oneflow_api.BlobDesc): A Blob
        y (oneflow_api.BlobDesc): A Blob. Must have the same type of x
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow_api.BlobDesc: A Blob, has the same type of x.

    For example:

    .. code-block:: python

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp

        @flow.global_function()
        def minimum_Job(x: tp.Numpy.Placeholder((3, )),
                        y: tp.Numpy.Placeholder((3, ))
        )->tp.Numpy:
            return flow.math.minimum(x, y)

        x = np.array([2, 3, 4]).astype(np.float32)
        y = np.array([4, 2, 1]).astype(np.float32)
        out = minimum_Job(x, y)

        # out [2. 2. 1.]

    """
    if x.shape == y.shape:
        return (
            flow.user_op_builder(name or id_util.UniqueStr("ElementWiseMinimum_"))
            .Op("elementwise_minimum")
            .Input("x", [x])
            .Input("y", [y])
            .Output("z")
            .Build()
            .InferAndTryRun()
            .RemoteBlobList()[0]
        )
    else:
        return build_broadcast_binary_op("broadcast_minimum", x, y, name)


@oneflow_export("math.maximum")
def maximum(
    x: oneflow_api.BlobDesc, y: oneflow_api.BlobDesc, name: Optional[str] = None
) -> oneflow_api.BlobDesc:
    """Returns the max of x and y element-wise, this op supports broadcasting.

    Args:
        x (oneflow_api.BlobDesc): A Blob
        y (oneflow_api.BlobDesc): A Blob. Must have the same type of x
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow_api.BlobDesc: A Blob, has the same type of x.

    For example:

    .. code-block:: python

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp

        @flow.global_function()
        def maximum_Job(x: tp.Numpy.Placeholder((3, )),
                        y: tp.Numpy.Placeholder((3, ))
        )->tp.Numpy:
            return flow.math.maximum(x, y)

        x = np.array([2, 3, 4]).astype(np.float32)
        y = np.array([4, 2, 1]).astype(np.float32)
        out = maximum_Job(x, y)

        # out [4. 3. 4.]

    """
    if x.shape == y.shape:
        return (
            flow.user_op_builder(name or id_util.UniqueStr("ElementWiseMaximum_"))
            .Op("elementwise_maximum")
            .Input("x", [x])
            .Input("y", [y])
            .Output("z")
            .Build()
            .InferAndTryRun()
            .RemoteBlobList()[0]
        )
    else:
        return build_broadcast_binary_op("broadcast_maximum", x, y, name)


@oneflow_export("math.reduced_shape_elem_cnt")
def elem_cnt(
    input_blob: oneflow_api.BlobDesc,
    axis: Optional[Sequence[int]] = None,
    dtype: Optional[flow.dtype] = None,
    name: Optional[str] = None,
) -> oneflow_api.BlobDesc:
    """Computes the product of input_blob's dimensions along the parameter `axis`. By default, all the dimensions will be computed.

    Args:
        input_blob (oneflow_api.BlobDesc): Input Blob
        axis (Optional[Sequence[int]], optional): The dimensions along which the op is performed. Defaults to None.
        dtype (Optional[flow.dtype], optional): The data type. Defaults to None.
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow_api.BlobDesc: A Blob

    For example:

    .. code-block:: python

        # Example 1:
        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp

        @flow.global_function()
        def elem_cnt_Job(x: tp.Numpy.Placeholder((3, 4, 5))
        )->tp.Numpy:
            return flow.math.reduced_shape_elem_cnt(x, axis=[0, 1])

        x = np.ones(shape=(3, 4, 5), dtype=np.float32)
        out = elem_cnt_Job(x) # 3 x 4 = 12

        # out [12]

        # Example 2:
        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp

        @flow.global_function()
        def elem_cnt_Job(x: tp.Numpy.Placeholder((3, 4, 5))
        )->tp.Numpy:
            return flow.math.reduced_shape_elem_cnt(x)

        x = np.ones(shape=(3, 4, 5), dtype=np.float32)
        out = elem_cnt_Job(x) # 3 x 4 x 5 = 60

        # out [60]

    """
    op_conf = op_conf_util.OperatorConf()
    setattr(
        op_conf,
        "name",
        name if name is not None else id_util.UniqueStr("ShapeElemCnt_"),
    )
    op_conf.shape_elem_cnt_conf.x = input_blob.unique_name
    if axis is None:
        op_conf.shape_elem_cnt_conf.exclude_axis_conf.SetInParent()
    else:
        assert isinstance(axis, (tuple, list))
        op_conf.shape_elem_cnt_conf.include_axis_conf.axis.extend(axis)
    if dtype is not None:
        op_conf.shape_elem_cnt_conf.data_type = oneflow_api.deprecated.GetProtoDtype4OfDtype(
            dtype
        )
    op_conf.shape_elem_cnt_conf.y = "y"
    interpret_util.Forward(op_conf)
    out_lbi = logical_blob_id_util.LogicalBlobId()
    out_lbi.op_name = op_conf.name
    out_lbi.blob_name = "y"
    return remote_blob_util.RemoteBlob(out_lbi)


def _top_k_at_last_dim(
    input: oneflow_api.BlobDesc,
    k: int = 1,
    sorted: bool = True,
    name: Optional[str] = None,
) -> oneflow_api.BlobDesc:
    return (
        flow.user_op_builder(name if name is not None else id_util.UniqueStr("TopK_"))
        .Op("top_k")
        .Input("in", [input])
        .Output("out")
        .Attr("k", k)
        .Attr("sorted", sorted)
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


@oneflow_export("math.top_k")
def top_k(
    input: oneflow_api.BlobDesc,
    axis: int = -1,
    k: int = 1,
    sorted: bool = True,
    name: Optional[str] = None,
) -> oneflow_api.BlobDesc:
    """Finds the indices of the k largest entries at specified axis, the difference between other framework is that oneflow only return the indices.

    Args:
        input (oneflow_api.BlobDesc): The input Blob
        axis (int, optional): dimension to be calculated. Defaults to the last dim (-1)
        k (int, optional): Number of top elements to look for along the last dimension. Defaults to 1.
        sorted (bool, optional): If true the resulting k elements will be sorted by the values in descending order. Defaults to True.
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow_api.BlobDesc: A Blob(dtype=int32) contains the indices of the k largest elements.

    For example:

    .. code-block:: python

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp

        @flow.global_function()
        def topk_Job(x: tp.Numpy.Placeholder((5, ))
        )->tp.Numpy:
            return flow.math.top_k(x, 2)

        x = np.array([1, 3, 8, 7, 2], dtype=np.float32)
        out = topk_Job(x)

        # out [2 3]

    """
    name = name if name is not None else id_util.UniqueStr("TopK_")
    num_axes = len(input.shape)
    axis = axis if axis >= 0 else axis + num_axes
    assert 0 <= axis < num_axes, "axis out of range"
    if axis == num_axes - 1:
        return _top_k_at_last_dim(input, k, sorted, name)
    else:
        perm = get_perm_when_transpose_axis_to_last_dim(num_axes, axis)
        x = flow.transpose(input, perm, False, True, name + "_transpose")
        x = _top_k_at_last_dim(x, k, sorted, name)
        return flow.transpose(
            x, get_inversed_perm(perm), False, True, name + "_inverse_transpose"
        )


def _argmax_at_last_dim(
    input: oneflow_api.BlobDesc, name: Optional[str] = None
) -> oneflow_api.BlobDesc:
    return (
        flow.user_op_builder(name if name is not None else id_util.UniqueStr("ArgMax_"))
        .Op("argmax")
        .Input("in", [input])
        .Output("out")
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


@oneflow_export("math.argmax")
def argmax(
    input: oneflow_api.BlobDesc, axis: int = -1, name: Optional[str] = None,
) -> oneflow_api.BlobDesc:
    """The op computes the index with the largest value of a Blob at specified axis.

    Args:
        input (oneflow_api.BlobDesc): Input Blob
        axis (int, optional): dimension to be calculated. Defaults to the last dim (-1)
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow_api.BlobDesc: A Blob(dtype=int32) contains the index with the largest value of `input`

    For example:

    .. code-block:: python

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp

        @flow.global_function()
        def argmax_Job(x: tp.Numpy.Placeholder((2, 5))
        )->tp.Numpy:
            return flow.math.argmax(x)

        x = np.array([[1, 3, 8, 7, 2],
                    [1, 9, 4, 3, 2]], dtype=np.float32)

        out = argmax_Job(x)

        # out [2 1]

    """
    name = name if name is not None else id_util.UniqueStr("ArgMax_")
    num_axes = len(input.shape)
    axis = axis if axis >= 0 else axis + num_axes
    assert 0 <= axis < num_axes, "axis out of range"
    if axis == num_axes - 1:
        return _argmax_at_last_dim(input, name)
    else:
        perm = get_perm_when_transpose_axis_to_last_dim(num_axes, axis)
        x = flow.transpose(input, perm, False, True, name + "_transpose")
        x = _argmax_at_last_dim(x, name)
        x = flow.expand_dims(x, -1, name + "_expand_dims")
        x = flow.transpose(
            x, get_inversed_perm(perm), False, True, name + "_inverse_transpose"
        )
        x = flow.squeeze(x, [axis], name + "_squeeze")
        return x


@oneflow_export("math.broadcast_to_compatible_with", "broadcast_to_compatible_with")
def broadcast_to_compatible_with(
    x: oneflow_api.BlobDesc,
    compatible: Sequence[oneflow_api.BlobDesc],
    name: Optional[str] = None,
) -> oneflow_api.BlobDesc:
    r"""Returns a 'Blob' with the shape can be broadcasted by other shapes

    Args:
        x (oneflow_api.BlobDesc): a 'Blob'
        compatible (Sequence[oneflow_api.BlobDesc]): Sequence of different shape
        name (Optional[str], optional): This operator's name. Defaults to None.

    Returns:
        oneflow_api.BlobDesc: A 'Blob' with the biggest shape

    For example:

    .. code-block:: python

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp

        @flow.global_function()
        def broadcast_to_compatible_with_Job(x: tp.Numpy.Placeholder((4, 1, 1))
        )->tp.Numpy:
            blob_a = flow.constant(value=1, dtype=flow.float32, shape=(1, 2, 1))
            blob_b = flow.constant(value=1, dtype=flow.float32, shape=(1, 1, 3))

            return flow.math.broadcast_to_compatible_with(x, [blob_a, blob_b])

        x = np.ones(shape=(4, 1, 1), dtype=np.float32)

        out = broadcast_to_compatible_with_Job(x)

        # out.shape (4, 2, 3)

    """
    assert isinstance(compatible, (list, tuple))
    if name is None:
        name = id_util.UniqueStr("BroadcastToCompatibleWith_")

    op_conf = op_conf_util.OperatorConf()
    setattr(op_conf, "name", name)
    setattr(op_conf.broadcast_to_compatible_with_conf, "x", x.unique_name)
    setattr(op_conf.broadcast_to_compatible_with_conf, "y", "y")
    op_conf.broadcast_to_compatible_with_conf.compatible.extend(
        [cp.unique_name for cp in compatible]
    )
    interpret_util.Forward(op_conf)

    ret_lbi = logical_blob_id_util.LogicalBlobId()
    ret_lbi.op_name = op_conf.name
    ret_lbi.blob_name = "y"
    return remote_blob_util.RemoteBlob(ret_lbi)


@oneflow_export(
    "math.clip_by_value", "clip_by_value", "clip_by_scalar", "clip", "clamp"
)
def clip_by_value(
    values: oneflow_api.BlobDesc,
    min_value: Optional[Union[int, float]] = None,
    max_value: Optional[Union[int, float]] = None,
    name: Optional[str] = None,
) -> oneflow_api.BlobDesc:
    """This op clips Blob values to a specified min value and max value.

    The equation is:

    .. math::
        out = MIN(MAX(x, min), max)

    Args:
        values (oneflow_api.BlobDesc): Input Blob
        min_value (Optional[Union[int, float]], optional): The minimum value to clip by. Defaults to None.
        max_value (Optional[Union[int, float]], optional): The maximum value to clip by. Defaults to None.
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Raises:
        ValueError: min_value and max_value `cannot be None at the same time`

    Returns:
        oneflow_api.BlobDesc: A clipped Blob

    For example:

    .. code-block:: python

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp

        @flow.global_function()
        def clip_by_value_Job(x: tp.Numpy.Placeholder((4, ))
        )->tp.Numpy:
            return flow.math.clip_by_value(x, min_value=-1, max_value=5)

        x = np.array([-2, 1, 4, 7], dtype=np.float32)

        out = clip_by_value_Job(x)

        # out [-1. 1. 4. 5.]

    """
    if name is None:
        name = id_util.UniqueStr("ClipByValue_")

    if min_value is not None and max_value is not None:
        op_builder = (
            flow.user_op_builder(name)
            .Op("clip_by_scalar")
            .Attr("floating_min", float(min_value))
            .Attr("integral_min", int(min_value))
            .Attr("floating_max", float(max_value))
            .Attr("integral_max", int(max_value))
        )
    elif min_value is not None:
        op_builder = (
            flow.user_op_builder(name)
            .Op("clip_by_scalar_min")
            .Attr("floating_min", float(min_value))
            .Attr("integral_min", int(min_value))
        )
    elif max_value is not None:
        op_builder = (
            flow.user_op_builder(name)
            .Op("clip_by_scalar_max")
            .Attr("floating_max", float(max_value))
            .Attr("integral_max", int(max_value))
        )
    else:
        raise ValueError("min_value and max_value cannot be None at the same time")

    return (
        op_builder.Input("x", [values])
        .Output("y")
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


@oneflow_export("math.l2_normalize")
def l2_normalize(
    input: oneflow_api.BlobDesc,
    axis: Optional[int] = None,
    epsilon: float = 1e-12,
    name: Optional[str] = None,
) -> oneflow_api.BlobDesc:
    r"""Use L2 norm to normalizes along dimension `axis`

    The equation is:

    .. math::
        out = \frac{x}{\sqrt{\Sigma{x^2}+\epsilon}}

    Args:
        input (oneflow_api.BlobDesc): Input Blob
        axis (Optional[int], optional): The axis on which to apply L2 normalization. Defaults to None.
        epsilon (float, optional): The epsilon value is used to avoid division by zero. Defaults to 1e-12.
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow_api.BlobDesc: The normalized Blob

    For example:

    .. code-block:: python

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp

        @flow.global_function()
        def l2_normalize_Job(x: tp.Numpy.Placeholder((4, ))
        )->tp.Numpy:
            return flow.math.l2_normalize(x, axis=0)

        x = np.array([1, 2, 3, 4], dtype=np.float32)

        out = l2_normalize_Job(x)

        # out [0.18257418 0.36514837 0.5477226  0.73029673]

    """
    if axis < 0:
        axis += len(input.shape)
    assert axis >= 0 and axis < len(input.shape)
    y, square_x_sum = (
        flow.user_op_builder(
            name if name is not None else id_util.UniqueStr("L2Normalize_")
        )
        .Op("l2_normalize")
        .Input("x", [input])
        .Output("y")
        .Output("square_x_sum")
        .Attr("axis", int(axis))
        .Attr("epsilon", float(epsilon))
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()
    )
    return y


@oneflow_export("math.squared_difference")
def squared_difference(
    x: Union[int, float, oneflow_api.BlobDesc],
    y: Union[int, float, oneflow_api.BlobDesc],
    name: Optional[str] = None,
) -> oneflow_api.BlobDesc:
    """This op computes :math:`(x - y)^2` element-wise.

    Args:
        x (Union[int, float, oneflow_api.BlobDesc]): A Blob
        y (Union[int, float, oneflow_api.BlobDesc]): A Blob with the same type of x
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow_api.BlobDesc: A Blob

    For example:

    .. code-block:: python

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp

        @flow.global_function()
        def squared_difference_Job(x: tp.Numpy.Placeholder((4, )),
                                y: tp.Numpy.Placeholder((4, ))
        )->tp.Numpy:
            return flow.math.squared_difference(x, y)

        x = np.array([1, 2, 3, 4], dtype=np.float32)
        y = np.array([2, 4, 6, 8], dtype=np.float32)

        out = squared_difference_Job(x, y)

        # out [ 1.  4.  9. 16.]

    """
    name_subtract, name_square = None, None
    if name is not None:
        name_subtract = name + "_subtract"
        name_square = name + "_square"
    return flow.math.square(flow.math.subtract(x, y, name_subtract), name_square)


@oneflow_export("math.gelu_grad")
def gelu_grad(
    x: oneflow_api.BlobDesc, dy: oneflow_api.BlobDesc, name: Optional[str] = None,
) -> oneflow_api.BlobDesc:
    return (
        flow.user_op_builder(
            name if name is not None else id_util.UniqueStr("GeluGrad_")
        )
        .Op("gelu_grad")
        .Input("x", [x])
        .Input("dy", [dy])
        .Output("dx")
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


@oneflow_export("math.tril", "nn.tril")
def tril(
    x: oneflow_api.BlobDesc,
    diagonal: int = 0,
    fill_value: Union[int, float] = 0,
    name: Optional[str] = None,
) -> oneflow_api.BlobDesc:
    r"""Compute lower triangle of an matrix.

    Args:
        x (oneflow_api.BlobDesc): Input Blob.
        diagonal (int): Diagonal offset, when diagonal > 0, diagonal offset up,
                        otherwise, offset downward.
        fill_value(Union[int, float]): The value filled into the upper triangle.
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Attention:
        The dimension of x must greater or equal to 2.

    Returns:
        oneflow_api.BlobDesc: The lower triangle blob of input.

    For example:

    .. code-block:: python

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp
        @flow.global_function()
        def tril_Job(x: tp.Numpy.Placeholder((4, 4))
        )->tp.Numpy:
            return flow.math.tril(x, 0)
        x = np.array([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]],
                      dtype=np.float32)
        out = tril_Job(x).get()

        # output [[1, 0, 0, 0],
                  [1, 2, 0, 0],
                  [1, 2, 3, 0],
                  [1, 2, 3, 4]]

    """
    if isinstance(fill_value, float):
        is_floating_fill_value = True
        floating_fill_value = float(fill_value)
        integer_fill_value = int(0)
    else:
        is_floating_fill_value = False
        floating_fill_value = float(0)
        integer_fill_value = int(fill_value)
    return (
        flow.user_op_builder(name if name is not None else id_util.UniqueStr("Tril_"))
        .Op("tril")
        .Input("in", [x])
        .Attr("diagonal", diagonal)
        .Attr("is_floating_fill_value", is_floating_fill_value)
        .Attr("floating_fill_value", floating_fill_value)
        .Attr("integer_fill_value", integer_fill_value)
        .Output("out")
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


@oneflow_export("math.fused_scale_tril", "nn.fused_scale_tril")
def fused_scale_tril(
    x: oneflow_api.BlobDesc,
    diagonal: int = 0,
    fill_value: Union[int, float] = 0,
    scale: Union[int, float] = 1,
    name: Optional[str] = None,
) -> oneflow_api.BlobDesc:

    if isinstance(fill_value, float):
        is_floating_fill_value = True
        floating_fill_value = float(fill_value)
        integer_fill_value = int(0)
    else:
        is_floating_fill_value = False
        floating_fill_value = float(0)
        integer_fill_value = int(fill_value)

    if isinstance(scale, float):
        is_floating_scale_value = True
        floating_scale_value = float(scale)
        integer_scale_value = int(1)
    else:
        is_floating_scale_value = False
        floating_scale_value = float(1)
        integer_scale_value = int(scale)
    return (
        flow.user_op_builder(
            name if name is not None else id_util.UniqueStr("FusedScaleTril_")
        )
        .Op("fused_scale_tril")
        .Input("in", [x])
        .Attr("diagonal", diagonal)
        .Attr("is_floating_fill_value", is_floating_fill_value)
        .Attr("floating_fill_value", floating_fill_value)
        .Attr("integer_fill_value", integer_fill_value)
        .Attr("is_floating_scale_value", is_floating_scale_value)
        .Attr("floating_scale_value", floating_scale_value)
        .Attr("integer_scale_value", integer_scale_value)
        .Output("out")
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


@oneflow_export("math.polyval")
def polyval(
    coeffs: Union[List, Tuple], x: oneflow_api.BlobDesc, name: Optional[str] = None
) -> oneflow_api.BlobDesc:
    r"""Computes the elementwise value of a polynomial.

    Args:
        coeffs (Union[List, Tuple]): The coefficients of the polynomial.
        x (oneflow_api.BlobDesc): A Blob.
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow_api.BlobDesc: A Blob, has the same data type of x.

    For example:

    .. code-block:: python

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp

        @flow.global_function()
        def polyval_Job(
            x: tp.Numpy.Placeholder((3,), dtype=flow.float32)
        ) -> tp.Numpy:
            coeffs = [1.0, 3.0, -2.0]
            return flow.math.polyval(coeffs, x)

        x = np.array([1.0, 2.0, 3.0]).astype(np.float32)
        out = polyval_Job(x)

        # output [ 2. 8. 16.]

    """
    if name is None:
        name = id_util.UniqueStr("Polyval_")
    if not isinstance(coeffs, (list, tuple)):
        raise ValueError(
            "Argument coeffs must be list type " "found {}".format(type(coeffs))
        )
    if len(coeffs) < 1:
        return flow.zeros_like(x, name=name)
    p = flow.zeros_like(x, name=name)
    for c in coeffs:
        p = flow.math.add(c, flow.math.multiply(p, x))
    return p


@oneflow_export("math.in_top_k", "in_top_k")
def in_top_k(
    targets: oneflow_api.BlobDesc,
    predictions: oneflow_api.BlobDesc,
    k: Optional[int],
    name: Optional[str] = None,
) -> oneflow_api.BlobDesc:
    r"""Says whether the targets are in the top K predictions.

    Args:
        targets (oneflow_api.BlobDesc): A Blob of type int32 or int64.
        predictions (oneflow_api.BlobDesc): A Blob of type float32.
        k (Optional[int], optional): Number of top elements to look at for computing precision.
        name (Optional[str], optional): The name for the operation. Defaults to None.
    Returns:
        oneflow_api.BlobDesc: A Blob of type bool. Computed Precision at k as a bool Blob.

    For example:

    .. code-block:: python

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp

        @flow.global_function()
        def intopk_Job(
            targets: tp.Numpy.Placeholder((2,), dtype=flow.int32),
            predictions: tp.Numpy.Placeholder((2, 4), dtype=flow.float32),
        ) -> tp.Numpy:
            return flow.math.in_top_k(targets, predictions, 1)

        targets = np.array([3, 1], dtype=np.int32)
        predictions = np.array([[0.0, 1.0, 2.0, 3.0], [3.0, 2.0, 1.0, 0.0],], dtype=np.float32)
        out = intopk_Job(targets, predictions)

        # out [1 0]

    """
    return (
        flow.user_op_builder(name if name is not None else id_util.UniqueStr("InTopK_"))
        .Op("in_top_k")
        .Input("targets", [targets])
        .Input("predictions", [predictions])
        .Attr("k", k)
        .Output("out")
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


@oneflow_export("range")
def range(start, limit=None, delta=1, dtype=None, name="range") -> oneflow_api.BlobDesc:
    r"""This operator is similar to python `range`, the difference is that `oneflow.range` generates
    a Blob.

    Args:
        start ([type]): The start of interval. Its type should be `int`.
        limit ([type], optional): The limit of interval. Its type should be `int`.
        delta (int, optional): The numerical spacing between elements. Defaults to 1.
        dtype ([type], optional): The output's data type. Currently we only support `oneflow.int64`. Defaults to None.
        name (str, optional): The name for the operation. Defaults to "range".

    Returns:
        oneflow_api.BlobDesc: The result Blob

    For example:

    Example 1:

    .. code-block:: python

        import oneflow as flow
        import oneflow.typing as tp


        @flow.global_function()
        def range_job()->tp.Numpy:
            with flow.scope.placement("cpu", "0:0"):
                out = flow.range(10, dtype=flow.int64)

            return out

        out = range_job()

        # out [0 1 2 3 4 5 6 7 8 9]

    Example2:

    .. code-block:: python

        import oneflow as flow
        import oneflow.typing as tp


        @flow.global_function()
        def range_job()->tp.Numpy:
            with flow.scope.placement("cpu", "0:0"):
                out = flow.range(1, 10, 3, dtype=flow.int64)

            return out

        out = range_job()

        # out [1 4 7]

    """
    # Ensure the dtype is not None
    assert dtype is not None, "Please specified data type"

    if limit is None:
        # If limit is None, We start from zero.
        start, limit = 0, start

    assert limit > start, "Limit should be larger than start"
    assert delta <= limit - start, "Delta is ilegal"

    # Ensure start, limit, delta's dtype is int, We will Add dtype hierarchy in Later version.
    assert type(start) == int, "Params `start`'s type should be int"
    assert type(limit) == int, "Params `limit`'s type should be int"
    assert type(delta) == int, "Params `delta`'s type should be int"

    # Build User OP
    return (
        flow.user_op_builder(name if name is not None else id_util.UniqueStr("Range_"))
        .Op("range")
        .Attr("start", start)
        .Attr("delta", delta)
        .Attr("limit", limit)
        .Attr("dtype", dtype)
        .Output("out")
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )
