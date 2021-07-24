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
from typing import Union

from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client.framework.tensor import Tensor, register_tensor_op
from oneflow.compatible.single_client.nn.module import Module


class Tile(Module):
    def __init__(self, reps: tuple) -> None:
        super().__init__()
        self.reps = reps

    def forward(self, input: Tensor) -> Tensor:
        reps = self.reps
        for s in self.reps:
            assert s > 0
        input_shape = input.shape
        diff = len(input_shape) - len(reps)
        if diff > 0:
            shape = [1 for _ in range(diff)]
            shape.extend([i for i in reps])
            reps = tuple(shape)
        return input.repeat(reps)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
