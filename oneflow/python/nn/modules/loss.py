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
    """

    def __init__(
        self, weight=None, ignore_index: int = None, reduction: str = "mean"
    ) -> None:
        super().__init__()
        if weight != None:
            raise ValueError("Argument weight is not supported yet")
        if ignore_index != None:
            raise ValueError("Argument ignore_index is not supported yet")
        assert reduction in [
            "sum",
            "none",
            None,
        ], "only 'sum' and None supported by now"

        self.reduction = reduction

        _opname = id_util.UniqueStr("Module_CrossEntropyLoss_")

        self._op = (
            flow.builtin_op("sparse_softmax_cross_entropy")
            .Name(_opname)
            .Input("prediction")
            .Input("label")
            .Output("prob")
            .Output("out")
        )

        self._reduce_sum_op = (
            flow.builtin_op("reduce_sum")
            .Name(_opname + "ReduceSum_")
            .Input("input_tensor")
            .Output("output_tensor")
        )

    def forward(self, input, target):
        self._op = self._op.Attr("depth", input.shape[len(input.shape) - 1]).Build()
        prob, out = self._op(input, target)
        if self.reduction == "mean":
            raise ValueError("not supported yet")
        elif self.reduction == "sum":
            self._reduce_sum_op = (
                self._reduce_sum_op.Attr("axis", list(range(len(out.shape))))
                .Attr("keepdims", False)
                .Build()
            )
            return self._reduce_sum_op(out)[0]
        else:
            return out
