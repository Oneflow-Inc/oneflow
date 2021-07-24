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
from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client.framework.tensor import register_tensor_op
from oneflow.compatible.single_client.nn.module import Module


class Unsqueeze(Module):
    def __init__(self, dim: int = 0) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, input):
        assert (
            -(1 + input.ndimension()) <= self.dim <= input.ndimension()
        ), "dim should within the range [-input.ndimension() - 1, input.ndimension() + 1)"
        if self.dim < 0:
            self.dim = 1 + input.ndimension() + self.dim
        return flow.F.expand_dims(input, axis=self.dim)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
