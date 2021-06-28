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
from oneflow.python.oneflow_export import oneflow_export, experimental_api
from oneflow.python.framework.tensor import register_tensor_op


class BMM(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input, mat2):
        return flow.F.batch_matmul(input, mat2)


@oneflow_export("bmm")
@register_tensor_op("bmm")
@experimental_api
def bmm_op(condition, x, y):
    """
    """
    return BMM()(condition, x, y)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
