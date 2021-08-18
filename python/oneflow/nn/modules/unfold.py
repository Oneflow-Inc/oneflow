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
from typing import Optional, Tuple, Union

import oneflow as flow
from oneflow.nn.common_types import _size_2_t
from oneflow.nn.module import Module
from oneflow.nn.modules.utils import _pair


class Unfold(Module):
    def __init__(
        self,
        kernel_size: _size_2_t,
        dilation: _size_2_t = 1,
        padding: _size_2_t = 0,
        stride: _size_2_t = 1
    ) -> None:
        super(Unfold, self).__init__()
        self.kernel_size = _pair(kernel_size)
        self.dilation = _pair(dilation)
        self.padding = _pair(padding)
        self.stride = _pair(stride)
        
        print("Python attr here!: ", self.kernel_size, self.dilation, self.padding, self.stride)

    def forward(self, input): 
        print("Python input here!: ", input.shape)
        return flow.F.unfold(input, "channels_first", self.kernel_size, self.dilation, self.padding, self.stride)

    def extra_repr(self) -> str:
        return 'kernel_size={kernel_size}, dilation={dilation}, padding={padding},' \
            ' stride={stride}'.format(**self.__dict__)

if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
