from typing import Optional, Sequence

import oneflow as flow
from oneflow.nn.module import Module


class TensorBufferToTensor(Module):
    def __init__(self, dtype, instance_shape):
        super().__init__()
        self._op = (
            flow.builtin_op("tensor_buffer_to_tensor")
            .Input("in")
            .Output("out")
            .Attr("dtype", dtype)
            .Attr("instance_shape", instance_shape)
            .Build()
        )

    def forward(self, input):
        return self._op(input)[0]


def tensor_buffer_to_tensor_op(x, dtype: flow.dtype, instance_shape: Sequence[int]):
    """This operator converts the Tensor's type from TensorBuffer to original type.
    Some operator's output data type is `TensorBuffer`, you can use this operator to convert back
    to `Tensor`.

    Refer to `Concept Explanation <https://docs.oneflow.org/basics_topics/concept_explanation.html#3tensorbuffer-tensorlist>`_
    for more about TensorBuffer.

    Args:
        x (oneflow.Tensor): The input Tensor.
        dtype (flow.dtype): The data dtype.
        instance_shape (Sequence[int]): The shape of each TensorBuffer instance.

    Returns:
        oneflow.Tensor: The result Tensor.

    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow
        >>> x = np.random.randn(4, 16, 64, 64).astype(np.float32)
        >>> x = flow.Tensor(x)
        >>> x = flow.tensor_to_tensor_buffer(x, instance_dims=2)
        >>> output = flow.tensor_buffer_to_tensor(x, instance_shape=(64, 64), dtype=flow.float)
        >>> output.shape
        flow.Size([4, 16, 64, 64])

    """
    return TensorBufferToTensor(dtype=dtype, instance_shape=instance_shape)(x)


class TensorToTensorBuffer(Module):
    def __init__(self, instance_dims):
        super().__init__()
        self._op = (
            flow.builtin_op("tensor_to_tensor_buffer")
            .Input("in")
            .Output("out")
            .Attr("instance_dims", instance_dims)
            .Build()
        )

    def forward(self, input):
        return self._op(input)[0]


def tensor_to_tensor_buffer(x, instance_dims: int):
    """This operator converts the Tensor's type to TensorBuffer.

    Refer to `Concept Explanation <https://docs.oneflow.org/basics_topics/concept_explanation.html#3tensorbuffer-tensorlist>`_
    for more about TensorBuffer.

    Args:
        x (oneflow.Tensor): The input Tensor.
        instance_dims (int): The dimensions of dynamic tensor instance.

    Returns:
        oneflow.Tensor: The result Tensor.

    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow
        >>> x = np.random.randn(4, 16, 64, 64).astype(np.float32)
        >>> x = flow.Tensor(x)
        >>> x = flow.tensor_to_tensor_buffer(x, instance_dims=2)
        >>> output = flow.tensor_buffer_to_tensor(x, instance_shape=(64, 64), dtype=flow.float)
        >>> output.shape
        flow.Size([4, 16, 64, 64])
    
    """
    return TensorToTensorBuffer(instance_dims=instance_dims)(x)


class GenTensorBuffer(Module):
    def __init__(self, shape, shape_list, value_list, data_type, dynamic_out):
        super().__init__()
        self._op = (
            flow.builtin_op("gen_tensor_buffer")
            .Output("out")
            .Attr("shape", shape)
            .Attr("shape_list", shape_list)
            .Attr("value_list", value_list)
            .Attr("data_type", data_type)
            .Attr("dynamic_out", dynamic_out)
            .Build()
        )

    def forward(self):
        return self._op()[0]


def gen_tensor_buffer(
    shape: Sequence[int],
    shape_list: Sequence[Sequence[int]],
    value_list: Sequence[float],
    data_type: Optional[flow.dtype] = flow.float32,
    dynamic_out: Optional[bool] = False,
):
    return GenTensorBuffer(shape, shape_list, value_list, data_type, dynamic_out)()


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
