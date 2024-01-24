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
from typing import List

from oneflow.nn.modules.module import Module
import oneflow.nn.functional as F


class AffineGrid(Module):
    def __init__(self, size: List[int], align_corners: bool = False):
        super().__init__()
        self.size = size
        self.align_corners = align_corners
    def forward(self, x):
        return F.affine_grid(x, self.size, self.align_corners)

if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
