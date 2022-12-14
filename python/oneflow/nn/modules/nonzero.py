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

import oneflow as flow
from oneflow.framework.tensor import register_tensor_op
from oneflow.nn.modules.module import Module


def nonzero_op(input, as_tuple=False):
    if as_tuple:
        return flow._C.nonzero(input, as_tuple)
    else:
        return flow._C.nonzero(input, as_tuple)[0]


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
