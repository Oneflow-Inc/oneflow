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


@register_tensor_op("unsqueeze")
def unsqueeze_op(input, dim):
    """Returns a new tensor with a dimension of size one inserted at the
    specified position.

    The returned tensor shares the same underlying data with this tensor.

    A :attr:`dim` value within the range `[-input.ndimension() - 1, input.ndimension() + 1)`
    can be used. Negative :attr:`dim` will correspond to :meth:`unsqueeze`
    applied at :attr:`dim` = ``dim + input.ndimension() + 1``.

    Args:
        input (Tensor): the input tensor.
        dim (int): the index at which to insert the singleton dimension

    For example: 

    .. code-block:: python 

        >>> import numpy as np
        >>> import oneflow as flow
        
        >>> x = flow.randn(2, 3, 4)
        >>> y = x.unsqueeze(2)
        >>> y.shape
        oneflow.Size([2, 3, 1, 4])
    """
    assert (
        -(1 + input.ndimension()) <= dim <= input.ndimension()
    ), f"Dim should within the range [{-input.ndimension() - 1}, {input.ndimension()}], but got {dim}"
    if dim < 0:
        dim = 1 + input.ndimension() + dim
    return flow._C.expand_dims(input, dim)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
