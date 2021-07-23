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
from oneflow.python.framework.tensor import register_tensor_op
from oneflow.python.nn.module import Module


class InTopk(Module):
    def __init__(self, k) -> None:
        super().__init__()
        self._in_top_k = (
            flow.builtin_op("in_top_k")
            .Input("targets")
            .Input("predictions")
            .Output("out")
            .Attr("k", k)
            .Build()
        )

    def forward(self, targets, predictions):
        assert (
            targets.shape[0] == predictions.shape[0]
        ), "The num of targets must equal the num of predictions"
        assert len(targets.shape) == 1, "The dimension of targets must be 1"
        assert len(predictions.shape) == 2, "The dimension of predictions must be 2"
        return self._in_top_k(targets, predictions)


@oneflow_export("in_top_k")
@experimental_api
def in_top_k_op(targets, predictions, k):
    r"""Says whether the targets are in the top K predictions.

    Args:
        targets (Tensor): the target tensor of type int32 or int64.
        predictions (Tensor): the predictions tensor of type float32 .
        k (int): Number of top elements to look at for computing precision.

    Returns:
        oneflow.Tensor: A Tensor of type bool. Computed Precision at k as a bool Tensor.

    For example:

    .. code-block:: python

        >>> import oneflow.experimental as flow
        >>> import numpy as np
        >>> flow.enable_eager_execution()
        >>> targets1 = flow.Tensor(np.array([3, 1]), dtype=flow.int32)
        >>> predictions1 = flow.Tensor(np.array([[0.0, 1.0, 2.0, 3.0], [3.0, 2.0, 1.0, 0.0],]), dtype=flow.float32)
        >>> out1 = flow.in_top_k(targets1, predictions1, k=1)
        >>> out1
        tensor([1, 0], dtype=oneflow.int8)
        >>> out2 = flow.in_top_k(targets1, predictions1, k=2)
        >>> out2
        tensor([1, 1], dtype=oneflow.int8)
        >>> targets2 = flow.Tensor(np.array([3, 1]), dtype=flow.int32, device=flow.device('cuda'))
        >>> predictions2 = flow.Tensor(np.array([[0.0, 1.0, 2.0, 3.0], [3.0, 2.0, 1.0, 0.0],]), dtype=flow.float32, device=flow.device('cuda'))
        >>> out3 = flow.in_top_k(targets2, predictions2, k=1)
        >>> out3
        tensor([1, 0], device='cuda:0', dtype=oneflow.int8)

    """

    return InTopk(k=k)(targets, predictions)[0]


@register_tensor_op("in_top_k")
@experimental_api
def in_top_k_op_tensor(targets, predictions, k):
    r"""

    in_top_k() -> Tensor

    See :func:`oneflow.experimental.in_top_k`

    """

    return InTopk(k=k)(targets, predictions)[0]


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
