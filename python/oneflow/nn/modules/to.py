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
import oneflow as flow
from oneflow.framework.tensor import register_tensor_op
import oneflow._oneflow_internal.lazy_mode as lazy_mode


@register_tensor_op("to")
def to_op(input, *args, **kwargs):
    """Performs Tensor dtype and/or device conversion.
        A flow.dtype and flow.device are inferred from the arguments of `input.to(*args, **kwargs)`.

    .. note::
        If the ``input`` Tensor already
        has the correct :class:`flow.dtype` and :class:`flow.device`, then ``input`` is returned.
        Otherwise, the returned tensor is a copy of ``input`` with the desired.

    Args:
        input (oneflow.Tensor): An input tensor.
        *args (oneflow.Tensor or oneflow.device or oneflow.dtype): Positional arguments
        **kwargs (oneflow.device or oneflow.dtype) : Key-value arguments

    Returns:
        oneflow.Tensor: A Tensor.

    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow

        >>> arr = np.random.randint(1, 9, size=(1, 2, 3, 4))
        >>> input = flow.Tensor(arr)
        >>> output = input.to(dtype=flow.float32)
        >>> np.array_equal(arr.astype(np.float32), output.numpy())
        True

    """

    return flow._C.to(input, *args, **kwargs)

    # device, dtype, copy = _parse_args(*args, **kwargs)

    # device, dtype = _validate_args(device, dtype, copy, input)

    # return flow._C.to(input, device, dtype, copy)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
