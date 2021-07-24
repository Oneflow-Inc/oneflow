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
from typing import List, Optional, Tuple

from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client.python.framework.tensor import (
    Tensor,
    register_tensor_op,
)
from oneflow.compatible.single_client.python.nn.module import Module


class Gather(Module):
    def __init__(self, dim: int = 0, sparse_grad: bool = False):
        super().__init__()
        assert sparse_grad is False, "Only support bool = False for now!"
        self.dim = dim

    def forward(self, input, index):
        assert self.dim < len(
            index.shape
        ), "Value of dim is out of range(dim should be less than len(index.shape))"
        assert len(input.shape) == len(
            index.shape
        ), "Dimensions of input and index should equal"
        for i in range(0, len(input.shape)):
            if self.dim == i:
                continue
            else:
                assert (
                    input.shape[i] == index.shape[i]
                ), "Dimensions of input and index should be same except at dim"
        return flow.F.dim_gather(input, index, dim=self.dim)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
