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
from oneflow.compatible.single_client.python.framework.tensor import register_tensor_op
from oneflow.compatible.single_client.python.nn.module import Module
from oneflow.compatible.single_client.python.oneflow_export import experimental_api


class TypeAs(Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        return input.to(dtype=target.dtype)


class Long(Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input.to(dtype=flow.int64)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
