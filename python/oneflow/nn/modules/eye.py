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


def eye_op(
    n, m=None, device: Union[str, flow.device] = "cpu", requires_grad: bool = False,
):
    """This operator creates a 2-D Tensor with ones on the diagonal and zeros elsewhere.

    Args:
        n (int): the number of rows.
        m (Optional[int], optional): the number of colums with default being n. Defaults to None.
    
    Keyword args:
        device(flow.device, optional): the desired device of returned tensor. Default: if None, uses the current device for the default tensor.
        requires_grad(bool, optional): If autograd should record operations on the returned tensor. Default: `False`.
    
    Returns:
        oneflow.Tensor: The result Blob with ones on the diagonal and zeros elsewhere.
    
    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> out = flow.eye(3, 3)
        >>> out
        tensor([[1., 0., 0.],
                [0., 1., 0.],
                [0., 0., 1.]], dtype=oneflow.float32)
    
    """
    if m is None:
        m = n

    if m == n:
        res = flow.diag(flow.ones(n))
    elif m < n:
        tmp = flow.ones(m)
        input1 = flow.zeros((n - m, m))
        input2 = flow.diag(tmp)
        res = flow.cat([input2, input1], dim=0)
    else:
        tmp = flow.ones(n)
        input1 = flow.zeros((n, m - n))
        input2 = flow.diag(tmp)
        res = flow.cat([input2, input1], dim=1)

    res.requires_grad = requires_grad
    if isinstance(device, str):
        device = flow.device(device)

    re = res.to(device)
    return re


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
