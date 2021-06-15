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
from oneflow.python.oneflow_export import oneflow_export, experimental_api
from oneflow.python.nn.module import Module


@oneflow_export("nn.L1Loss")
@experimental_api
class L1Loss(Module):
    r"""This operator computes the L1 Loss between each element in `input` and `target`.

    The equation is:

    if reduction = "none":

    .. math::

        output = |Target - Input|

    if reduction = "mean":

    .. math::

        output = \frac{1}{n}\sum_{i=1}^n|Target_i - Input_i|

    if reduction = "sum":

    .. math::

        output = \sum_{i=1}^n|Target_i - Input_i|

    Args:
        input (oneflow.experimental.Tensor): The input Tensor.
        target (oneflow.experimental.Tensor): The target Tensor.
        reduction (str): The reduce type, it can be one of "none", "mean", "sum". Defaults to "mean".

    Returns:
        oneflow.experimental.Tensor: The result Tensor.

    For example:

    .. code-block:: python

        >>> import oneflow.experimental as flow
        >>> import numpy as np
        >>> flow.enable_eager_execution()

        >>> input = flow.Tensor([[1, 1, 1], [2, 2, 2], [7, 7, 7]], dtype = flow.float32)
        >>> target = flow.Tensor([[4, 4, 4], [4, 4, 4], [4, 4, 4]], dtype = flow.float32)
        >>> m = flow.nn.L1Loss(reduction="none")
        >>> out = m(input, target)
        >>> out
        tensor([[3., 3., 3.],
                [2., 2., 2.],
                [3., 3., 3.]], dtype=oneflow.float32)
        >>> m_mean = flow.nn.L1Loss(reduction="mean")
        >>> out = m_mean(input, target)
        >>> out
        tensor([2.6667], dtype=oneflow.float32)
        >>> m_mean = flow.nn.L1Loss(reduction="sum")
        >>> out = m_mean(input, target)
        >>> out
        tensor([24.], dtype=oneflow.float32)
        
    """

    def __init__(self, reduction: str = "mean", reduce=True) -> None:
        super().__init__()
        if reduce is not None and not reduce:
            raise ValueError("Argument reduce is not supported yet")
        assert reduction in [
            "none",
            "mean",
            "sum",
            None,
        ], "only 'sum', 'mean' and 'none' supported by now"

        self.reduction = reduction

    def forward(self, input, target):
        assert (
            input.shape == target.shape
        ), "The Input shape must be the same as Target shape"

        l1_value = flow.experimental.abs(flow.experimental.sub(target, input))
        if self.reduction == "mean":
            return flow.experimental.mean(l1_value)
        elif self.reduction == "sum":
            return flow.experimental.sum(l1_value)
        else:
            return l1_value


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
