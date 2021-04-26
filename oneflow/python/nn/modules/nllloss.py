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
from oneflow.python.oneflow_export import oneflow_export
from oneflow.python.framework.tensor import register_tensor_op_by_module
from oneflow.python.framework.tensor import register_op_by_module
import numpy as np


@oneflow_export("nn.NLLLoss")
class NLLLoss(Module):
    def __init__(
        self, weight=None, ignore_index: int = None, reduction: str = "mean",
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

    def forward(self, input, target):
        n = input.shape[0]
        c = input.shape[1]
        input = flow.negative(input)
        mask = np.array(target[0:n].numpy())

        input = [input[i, int(mask[i]),].numpy() for i in range(n)]
        # print(input[0].numpy())

        if self.reduction == "sum":
            loss = 0
            for x in input:
                loss += x
            return loss
        elif self.reduction == "mean":
            loss = 0
            for x in input:
                loss += x
            return loss / n
        else:
            return flow.cat(input)
