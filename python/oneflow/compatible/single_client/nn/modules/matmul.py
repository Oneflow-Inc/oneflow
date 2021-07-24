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
from typing import Optional, Sequence

from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client.framework import id_util as id_util
from oneflow.compatible.single_client.framework.tensor import register_tensor_op
from oneflow.compatible.single_client.nn.module import Module


class MatMul(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, a, b):
        assert len(a.shape) >= 2, "Tensor a's dim should >=2"
        assert len(b.shape) >= 2, "Tensor b's dim should >=2"
        if len(a.shape) == len(b.shape):
            if len(a.shape) == 2:
                res = flow.F.matmul(a, b)
            else:
                res = flow.F.batch_matmul(a, b)
        else:
            assert (
                len(b.shape) == 2
            ), "Not support number of dimensions of a being less than number of dimensions of b!"
            res = flow.F.broadcast_matmul(a, b)
        return res


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
