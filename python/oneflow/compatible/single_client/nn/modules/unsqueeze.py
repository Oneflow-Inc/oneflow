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


class Unsqueeze(Module):
    def __init__(self, dim: int = 0) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, input):
        assert (
            -(1 + input.ndimension()) <= self.dim <= input.ndimension()
        ), "dim should within the range [-input.ndimension() - 1, input.ndimension() + 1)"
        if self.dim < 0:
            self.dim = 1 + input.ndimension() + self.dim
        return flow.F.expand_dims(input, axis=self.dim)


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
        >>> import oneflow.compatible.single_client.experimental as flow
        >>> flow.enable_eager_execution()

        >>> x = flow.Tensor(np.random.rand(2, 3, 4))
        >>> y = x.unsqueeze(2)
        >>> y.shape
        flow.Size([2, 3, 1, 4])
    """
    return Unsqueeze(dim)(input)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
