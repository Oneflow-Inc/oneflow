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

from typing import Optional, Union

import oneflow as flow

from oneflow.nn.common_types import _size_any_t
from oneflow.nn.modules.utils import _single


def empty_op(
    size: Union[_size_any_t, flow.Size],
    dtype: Optional[flow.dtype] = None,
    device: Union[flow.device, str] = None,
    requires_grad: bool = False,
):
    """
    Returns a tensor filled with uninitialized data.
    The shape of the tensor is defined by the variable argument ``size``.

    Args:
        size (an integer or tuple of integer values) – Defining the shape of the output tensor. Can be
          a variable number of arguments or a collection like a list or tuple.
        dtype (flow.dtype, optional) - The desired data type of returned tensor. Default: ``flow.float32``.
        device (torch.device, optional) – The desired device of returned tensor. Default: if None, uses the
          current device for the default tensor type
        requires_grad (bool, optional) – If autograd should record operations on the returned tensor. Default: False.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> y = flow.empty(5)
        >>> y.shape
        flow.Size([5])

    """
    assert size is not None, "shape must not be None"
    assert isinstance(
        size, (int, tuple, flow.Size)
    ), "shape should be int or tuple of int or flow.Size"

    shape = _single(size)
    if dtype is None:
        dtype = flow.float32
    if device is None:
        device = flow.device("cpu")

    tensor = flow.F.empty(shape=shape, dtype=dtype)
    tensor = tensor.to(device=device)
    tensor.requires_grad_(requires_grad)
    return tensor


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
