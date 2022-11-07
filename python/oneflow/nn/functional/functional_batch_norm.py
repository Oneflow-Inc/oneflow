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


def batch_norm(
    input: Tensor,
    running_mean: Optional[Tensor],
    running_var: Optional[Tensor],
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    training: bool = False,
    momentum: float = 0.1,
    eps: float = 1e-5,
) -> Tensor:
    r"""Applies Batch Normalization for each channel across a batch of data.

    See :class:`~oneflow.nn.BatchNorm1d`, :class:`~oneflow.nn.BatchNorm2d`,
    :class:`~oneflow.nn.BatchNorm3d` for details.
    """
    if input.ndim == 4 and os.getenv("ONEFLOW_ENABLE_NHWC") == "1":
        axis = 3
    else:
        axis = 1

    return flow._C.normalization(
        input, running_mean, running_var, weight, bias, axis, eps, momentum, training,
    )
