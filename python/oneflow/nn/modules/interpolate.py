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
import math
import warnings
from typing import Optional, Tuple, Union

import oneflow as flow
from oneflow.framework.tensor import register_tensor_op
from oneflow.nn.modules.module import Module
import oneflow.nn.functional as F


class Interpolate(Module):
    def __init__(
        self,
        size: Optional[Union[int, Tuple[int, ...]]] = None,
        scale_factor: Optional[Union[float, Tuple[float, ...]]] = None,
        mode: str = "nearest",
        align_corners: Optional[bool] = None,
        recompute_scale_factor: Optional[bool] = None,
    ):
        super().__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
        self.recompute_scale_factor = recompute_scale_factor

    def forward(self, x):
        return F.interpolate(
            x,
            self.size,
            self.scale_factor,
            self.mode,
            self.align_corners,
            self.recompute_scale_factor,
        )


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
