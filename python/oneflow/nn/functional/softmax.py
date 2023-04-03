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
import os
from typing import List, Optional

from oneflow.framework.tensor import Tensor
import oneflow as flow

# ref https://github.com/pytorch/pytorch/blob/master/torch/nn/functional.py
def softmax(input: Tensor, dim: Optional[int] = None, _stacklevel: int = 3, dtype = None) -> Tensor:
    r"""Applies a softmax function.
    Softmax is defined as:
    :math:`\text{Softmax}(x_{i}) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}`
    It is applied to all slices along dim, and will re-scale them so that the elements
    lie in the range `[0, 1]` and sum to 1.
    See :class:`~torch.nn.Softmax` for more details.
    Args:
        input (Tensor): input
        dim (int): A dimension along which softmax will be computed.
        dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
          If specified, the input tensor is casted to :attr:`dtype` before the operation
          is performed. This is useful for preventing data type overflows. Default: None.
    .. note::
        This function doesn't work directly with NLLLoss,
        which expects the Log to be computed between the Softmax and itself.
        Use log_softmax instead (it's faster and has better numerical properties).
    """
    if dtype is None:
        ret = flow._C.softmax(input, dim)
    else:
        ret = flow._C.softmax(input.to(dtype), dim)
    return ret