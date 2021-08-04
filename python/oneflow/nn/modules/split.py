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
from typing import Union, List
import numpy as np

import oneflow as flow
from oneflow.framework.tensor import Tensor, register_tensor_op
from oneflow.nn.module import Module


class Split(Module):
    def __init__(
        self, split_size_or_sections: Union[int, List[int]], dim: int = 0
    ) -> None:
        super().__init__()
        self.split_size_or_sections = split_size_or_sections
        self.dim = dim

    def forward(self, x):
        dim = dim + x.dim if self.dim < 0 else self.dim
        if isinstance(self.split_size_or_sections, list):
            return tuple(
                flow.F.split_with_size(
                    x, split_sizes=self.split_size_or_sections, dim=dim
                )
            )
        else:
            return tuple(
                flow.F.split(x, split_size=self.split_size_or_sections, dim=dim)
            )


@register_tensor_op("split")
def split_op(x, split_size_or_sections: Union[int, List[int]], dim: int = 0):
    return Split(split_size_or_sections, dim)(x)
