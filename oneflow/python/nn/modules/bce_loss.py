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

from typing import Optional
import oneflow as flow
from oneflow.python.oneflow_export import oneflow_export, experimental_api
from oneflow.python.nn.module import Module
from oneflow.python.framework.tensor import register_tensor_op
from oneflow.python.nn.modules.math_ops import Sum, Mean, Log
from oneflow.python.nn.modules.negative import Negative


@oneflow_export("nn.BCELoss")
@experimental_api
class BCELoss(Module):
    r"""This operator computes the binary cross entropy loss.

    The equation is:

    if reduction = "none":

    .. math::

        out = -(Target_i*log(Input_i) + (1-Target_i)*log(1-Input_i))

    if reduction = "mean":

    .. math::

        out = -\frac{1}{n}\sum_{i=1}^n(Target_i*log(Input_i) + (1-Target_i)*log(1-Input_i))

    if reduction = "sum":

    .. math::

        out = -\sum_{i=1}^n(Target_i*log(Input_i) + (1-Target_i)*log(1-Input_i))

    Args:
        input (oneflow.experimental.Tensor): The input Tensor.
        target (oneflow.experimental.Tensor): The target Tensor.
        weight (oneflow.experimental.Tensor, optional): The manual rescaling weight to the loss. Default to None, whose corresponding weight value is 1.
        reduction (str, optional): The reduce type, it can be one of "none", "mean", "sum". Defaults to "mean".

    Attention:
        The input value must be in the range of (0, 1). Or the loss function may return `nan` value.

    Returns:
        oneflow._oneflow_internal.BlobDesc: The result Blob.

    For example:

    .. code-block:: python

        import oneflow as flow
        import oneflow.typing as tp
        import numpy as np


        >>> import oneflow.experimental as flow
        >>> import numpy as np
        >>> flow.enable_eager_execution()

        >>> input = flow.Tensor(np.array([[1.2, 0.2, -0.3], [0.7, 0.6, -2]]).astype(np.float32))
        >>> target = flow.Tensor(np.array([[0, 1, 0], [1, 0, 1]]).astype(np.float32))
        >>> weight = flow.Tensor(np.array([[2, 2, 2], [2, 2, 2]]).astype(np.float32))
        >>> activation = flow.nn.Sigmoid()
        >>> sigmoid_input = activation(input)
        >>> m = flow.nn.BCELoss(reduction="none")
        >>> out = m(sigmoid_input, target, weight)
        >>> out
        [2.0611262]
        
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
        self.mean = Mean()
        self.sum = Sum()
        self.negative = Negative()
        self.log = Log()


    def forward(self, input, target, weight):
        assert (
            input.shape == target.shape
        ), "The Input shape must be the same as Target shape"

        _cross_entropy_loss = self.negative(
            target * self.log(input) + (1 - target) * self.log(1 - input)
            )

        if weight is not None:
            assert (
                weight.shape == input.shape
            ), "The weight shape must be the same as Input shape"
            _weighted_loss = weight * _cross_entropy_loss
        else:
            _weighted_loss = _cross_entropy_loss

        if self.reduction == "mean":
            return self.mean(_weighted_loss)
        elif self.reduction == "sum":
            return self.sum(_weighted_loss)
        else:
            return _weighted_loss


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=False)
