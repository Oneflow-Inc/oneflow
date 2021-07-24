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

from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client.python.framework.tensor import register_tensor_op
from oneflow.compatible.single_client.python.nn.module import Module


class Expand(Module):
    def __init__(self, *sizes) -> None:
        super().__init__()
        self.expand_size = list(*sizes)

    def forward(self, x):
        if x.dtype == flow.int8:
            x = flow.experimental.cast(x, flow.int32)
        expand_size = self.expand_size
        assert len(expand_size) >= len(
            x.shape
        ), "The desired expanded dims should not be less than the input dims."
        original_stride = [1]
        for i in range(len(x.shape) - 2, -1, -1):
            original_stride.insert(0, original_stride[0] * x.shape[i + 1])
        new_size = []
        new_stride = []
        diff = len(expand_size) - len(x.shape)
        for i in range(len(expand_size) - 1, -1, -1):
            if i >= diff:
                if expand_size[i] == -1 or expand_size[i] == x.shape[i - diff]:
                    new_size.insert(0, x.shape[i - diff])
                    new_stride.insert(0, original_stride[i - diff])
                else:
                    assert expand_size[i] >= 1 and x.shape[i - diff] == 1
                    new_size.insert(0, expand_size[i])
                    new_stride.insert(0, 0)
            else:
                assert expand_size[i] >= 1
                new_size.insert(0, expand_size[i])
                if expand_size[i] == 1:
                    new_stride.insert(0, new_stride[0])
                else:
                    new_stride.insert(0, 0)
        return flow.F.expand(
            x, in_shape=list(x.shape), out_shape=new_size, stride=new_stride
        )


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
