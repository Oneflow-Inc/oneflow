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
from typing import Optional

import numpy as np

from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client.python.framework.tensor import register_tensor_op
from oneflow.compatible.single_client.python.nn.module import Module


class Argwhere(Module):
    def __init__(self, dtype) -> None:
        super().__init__()
        if dtype == None:
            dtype = flow.int32
        self.dtype = dtype

    def forward(self, x):
        (res, size) = flow.F.argwhere(x, dtype=self.dtype)
        slice_tup_list = [[0, int(size.numpy()), 1]]
        return flow.experimental.slice(res, slice_tup_list=slice_tup_list)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
