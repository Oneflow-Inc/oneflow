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

from oneflow.python.oneflow_export import oneflow_export
from oneflow.python.nn.module import Module
from oneflow.python.nn.modules.utils import (
    _single,
    _pair,
    _triple,
    _reverse_repeat_tuple,
)
from oneflow.python.nn.common_types import _size_1_t, _size_2_t, _size_3_t
from typing import Optional, List, Tuple
from oneflow.python.ops.nn_ops import calc_pool_padding, get_dhw_offset
import oneflow.python.framework.id_util as id_util


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

    Shape:
        - Input: :math:`(N, C)` where `C = number of classes`, or
          :math:`(N, C, d_1, d_2, ..., d_K)` with :math:`K \geq 1`
          in the case of `K`-dimensional loss.
        - Target: :math:`(N)` where each value is :math:`0 \leq \text{targets}[i] \leq C-1`, or
          :math:`(N, d_1, d_2, ..., d_K)` with :math:`K \geq 1` in the case of
          K-dimensional loss.
        - Output: scalar.
          If :attr:`reduction` is ``'none'``, then the same size as the target:
          :math:`(N)`, or
          :math:`(N, d_1, d_2, ..., d_K)` with :math:`K \geq 1` in the case
          of K-dimensional loss.

    For example:

    .. code-block:: python 

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
        ignore_index: int = None,
        reduction: str = "mean",
        name: Optional[str] = None,
    ) -> None:
        super().__init__()
        if weight != None:
            raise ValueError("Argument weight is not supported yet")
        if ignore_index != None:
            raise ValueError("Argument ignore_index is not supported yet")
        assert reduction in [
            "sum",
            "none",
            "mean",
            None,
        ], "only 'sum', 'mean' and None supported by now"

        self.reduction = reduction

        self._op = (
            flow.builtin_op("sparse_softmax_cross_entropy", name)
            .Input("prediction")
            .Input("label")
            .Output("prob")
            .Output("out")
        )

    def forward(self, input, target):
        self._op = self._op.Attr("depth", input.shape[len(input.shape) - 1]).Build()
        prob, out = self._op(input, target)
        if self.reduction == "mean":
            self._mean = flow.Mean()
            return self._mean(out)
        elif self.reduction == "sum":
            self._sum = flow.Sum()
            return self._sum(out)
        else:
            return out


# TODO(Liang Depeng): just for resnet50, will be removed after refactoring the CrossEntropyLoss
@oneflow_export("nn.CrossEntropyLossV2")
class CrossEntropyLossV2(Module):
    def __init__(self, reduction: str = "mean",) -> None:
        super().__init__()
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
            .Attr("depth", 1000)
            .Output("prob")
            .Output("out")
            .Build()
        )

        self._sum_op = (
            flow.builtin_op("reduce_sum")
            .Input("input_tensor")
            .Output("output_tensor")
            .Attr("keepdims", False)
            .Attr("axis", [0,])
            .Build()
        )

    def forward(self, input, target):
        prob, out = self._op(input, target)
        sums = self._sum_op(out)[0]
        reduce_count = 1
        for dim in out.shape:
            reduce_count *= dim
        return flow.mul(sums, 1.0 / reduce_count)
