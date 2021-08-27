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
from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client.framework.tensor import register_tensor_op
from oneflow.compatible.single_client.nn.module import Module


class Negative(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return flow.F.negative(x)


@register_tensor_op("negative")
def negative_op(x):
    """This operator computes the negative value of Tensor.

    Args:
        x (oneflow.compatible.single_client.Tensor): A Tensor

    Returns:
        oneflow.compatible.single_client.Tensor: The result Tensor

    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow.compatible.single_client.experimental as flow
        >>> flow.enable_eager_execution()

        >>> input = flow.Tensor(
        ...    np.array([1.0, -1.0, 2.3]).astype(np.float32), dtype=flow.float32
        ... )
        >>> out = flow.negative(input)
        >>> out
        tensor([-1. ,  1. , -2.3], dtype=oneflow.float32)

    """
    return Negative()(x)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
