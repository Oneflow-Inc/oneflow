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
from oneflow.python.nn.module import Module
from oneflow.python.oneflow_export import oneflow_export, experimental_api
from oneflow.python.framework.tensor import register_tensor_op
import oneflow.python.framework.id_util as id_util
from typing import Optional, Sequence


class Squeeze(Module):
    def __init__(self, dim: Optional[Sequence[int]] = None) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x):
        if self.dim is None:
            return x
        return flow.F.squeeze(x, dim=self.dim)


@oneflow_export("squeeze")
@register_tensor_op("squeeze")
@experimental_api
def squeeze_op(input, dim: Optional[Sequence[int]] = None):
    """This operator removes the specified dimention which size is 1 of the input Tensor.
    If the `dim` is not specified, this operator will remove all the dimention which size is 1 of the input Tensor.

    The amount of element in return value is the same as Tensor `input`.

    Args:
        input (oneflow.Tensor): The input Tensor.
        dim (Optional[Sequence[int]]): The dim. Defaults to None.

    Returns:
        Tensor: The result Tensor.

    For example:

    .. code-block:: python

        >>> import oneflow.experimental as flow
        >>> import numpy as np
        >>> flow.enable_eager_execution()

        >>> input = flow.Tensor(np.array([[[[1, 1, 1]]]]).astype(np.int32))
        >>> out = flow.squeeze(input, dim=[1, 2]).numpy().shape
        >>> print(out)
        (1, 3)

    """
    if isinstance(dim, int):
        dim = [dim]
    elif dim is None:
        dim = range(input.ndim)

    dim = list(filter(lambda i: input.size(i) == 1, dim))
    return Squeeze(dim=dim)(input)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
