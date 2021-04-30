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
import numpy as np

@oneflow_export("nn.NLLLoss")
class NLLLoss(Module):
    r""" The negative log likelihood loss. It is useful to train a classification problem with C classes.
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
        
        import oneflow as flow
        import numpy as np

        input = flow.Tensor(
            [[-0.1664078, -1.7256707, -0.14690138],
                [-0.21474946, 0.53737473, 0.99684894],
                [-1.135804, -0.50371903, 0.7645404]], dtype=flow.float32)
        target = flow.Tensor(np.array([0, 1, 2]), dtype=flow.int32)
        out = flow.nn.NLLLoss(reduction="none")(input, target)
        # out: [0.80199665 1.1166505  0.35826027]

        out_sum = flow.nn.NLLLoss(reduction="sum")(input, target)
        # out_sum: [2.2769074]
        
        out_mean = flow.nn.NLLLoss(reduction="mean")(input, target)
        # out_mean: [0.7589692]
    
    """

    def __init__(
        self, weight=None, ignore_index: int = None, reduction: str = "none",
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
        self._gather_nd_op = (
            flow.builtin_op("gather_nd")
            .Input("params")
            .Input("indices")
            .Output("out")
            .Build()
        )

    def forward(self, input, target):
        n = input.shape[0]
        idx = flow.unsqueeze(flow.arange(0, b, 1), dim=1)
        target = flow.unsqueeze(target, axis=1)
        t = flow.cat([idx, target], axis=1)
        res = self._gather_nd_op(x, indices=t)[0]
        if self.reduction == 'none':
            return res
        elif self.reduction == 'sum':
            return flow.sum(res)
        else:
            return flow.mean(res)


        
