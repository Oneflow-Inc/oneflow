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
from typing import Union

import oneflow as flow
from oneflow.framework.tensor import register_tensor_op


def is_floating_point(input):
    if input.dtype in (flow.float, flow.float16, flow.float32, flow.float64):
        return True
    return False


@register_tensor_op("cpu")
def cpu(input):
    r"""Returns a copy of this object in CPU memory.
    If this object is already in CPU memory and on the correct device, then no copy is performed and the original object is returned.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        
        >>> input = flow.tensor([1, 2, 3, 4, 5], device=flow.device("cuda"))
        >>> output = input.cpu()
        >>> output.device
        device(type='cpu', index=0)
    """
    return input.to(device="cpu")


@register_tensor_op("cuda")
def cuda(input, device: Union[int, str, flow.device] = None):
    r"""Returns a copy of this object in CUDA memory.
    If this object is already in CUDA memory and on the correct device, then no copy is performed and the original object is returned.

    Args:
        device  (flow.device): The destination GPU device. Defaults to the current CUDA device.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        
        >>> input = flow.Tensor([1, 2, 3, 4, 5])
        >>> output = input.cuda()
        >>> output.device
        device(type='cuda', index=0)
    """
    if device is None:
        device = "cuda"
    elif device is isinstance(int):
        device = "cuda:" + str(device)
    return input.to(device=device)


@register_tensor_op("item")
def item_op(input):
    r"""Returns the value of this tensor as a standard Python number. This only works for tensors with one element. 
    For other cases, see tolist().

    This operation is not differentiable.

    Args:
        input  (Tensor): the input tensor.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> x = flow.tensor([1.0])
        >>> x.item()
        1.0
    """
    assert input.numel() == 1, "Only a Tensor with 1 element can be converted to Scalar"
    return input.numpy().item()


@register_tensor_op("tolist")
def tolist_op(input):
    r"""Returns the tensor as a (nested) list. For scalars, a standard Python number is returned, 
    just like with `item()`. Tensors are automatically moved to the CPU first if necessary.

    This operation is not differentiable.

    Args:
        input  (Tensor): the input tensor.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> input = flow.tensor([[1,2,3], [4,5,6]])
        >>> input.tolist()
        [[1, 2, 3], [4, 5, 6]]
    """
    if input.numel() == 1 and input.ndim == 0:
        return input.item()
    return input.numpy().tolist()


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
