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
import oneflow
from oneflow.framework.docstr.utils import add_docstr

add_docstr(
    oneflow._C.randn,
    """
    Returns a tensor filled with random numbers from a normal distribution with mean 0 and variance 1 (also called the standard normal distribution).

    The shape of the tensor is defined by the variable argument ``size``.

    Args:
        size (int... or oneflow.Size): Defining the shape of the output tensor.
          Can be a variable number of arguments or a collection like a list or tuple or oneflow.Size.
        out (optional): The output tensor.
        dtype (flow.dtype, optional): The desired data type of returned tensor. Default: ``flow.float32``.
        layout (optional): The desired layout of returned Tensor.
        generator (flow.Generator, optional): a pseudorandom number generator for sampling
        device (flow.device, optional): The desired device of returned local tensor. If None, uses the
          current device.
        placement (flow.placement, optional): The desired device of returned global tensor. If None, will
          construct local tensor.
        sbp (flow.sbp, optional): The desired sbp of returned global tensor. It must be equal with the
          numbers of placement.
        requires_grad (bool, optional): If autograd should record operations on the returned tensor. Default: False.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> x = flow.randn(3,3) # construct local tensor
        >>> x.shape
        oneflow.Size([3, 3])
        >>> x.is_global
        False
        >>> placement = flow.placement("cpu", ranks=[0])
        >>> sbp = flow.sbp.broadcast
        >>> x = flow.randn(3,3,placement=placement,sbp=sbp) # construct global tensor
        >>> x.is_global
        True

    """
)
