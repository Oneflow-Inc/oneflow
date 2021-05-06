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
from oneflow.python.oneflow_export import oneflow_export
from oneflow.python.nn.module import Module


@oneflow_export("nn.CrossEntropyLoss")
class CrossEntropyLoss(Module):
    r"""
    Args:
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will
            be applied, ``'mean'``: the weighted mean of the output is taken,
            ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in
            the meantime, specifying either of those two args will override
            :attr:`reduction`. Default: ``'mean'``

    For example:
    .. code-block:: python

        import oneflow as flow
        input = flow.Tensor(
            [[-0.1664078, -1.7256707, -0.14690138],
                [-0.21474946, 0.53737473, 0.99684894],
                [-1.135804, -0.50371903, 0.7645404]], dtype=flow.float32)
        target = flow.Tensor(np.array([0, 1, 2]), dtype=flow.int32)
        out = flow.nn.CrossEntropyLoss(reduction="none")(input, target)
        # out: [0.80199665 1.1166505  0.35826027]
        out_sum = flow.nn.CrossEntropyLoss(reduction="sum")(input, target)
        # out_sum: [2.2769074]
        out_mean = flow.nn.CrossEntropyLoss(reduction="mean")(input, target)
        # out_mean: [0.7589692]

    """

    def __init__(
        self,
        weight=None,
        ignore_index: Optional[int] = None,
        reduction: Optional[str] = "mean",
    ) -> None:
        super().__init__()
        if weight is not None:
            raise ValueError("Argument weight is not supported yet")
        if ignore_index is not None:
            raise ValueError("Argument ignore_index is not supported yet")
        assert reduction in [
            "sum",
            "none",
            "mean",
            None,
        ], "only 'sum', 'mean' and None supported by now"

        self.reduction = reduction
        self._op = (
            flow.builtin_op("sparse_softmax_cross_entropy")
            .Input("prediction")
            .Input("label")
            .Output("prob")
            .Output("out")
            .Build()
        )

    def forward(self, input, target):
        prob, out = self._op(input, target, depth=input.shape[len(input.shape) - 1])
        if self.reduction == "mean":
            return flow.mean(out)
        elif self.reduction == "sum":
            return flow.sum(out)
        else:
            return out
