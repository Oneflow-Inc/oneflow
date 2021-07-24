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
import random
import sys
import traceback
from typing import List, Optional, Sequence, Tuple, Union

from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client.nn.common_types import (
    _size_1_t,
    _size_2_t,
    _size_3_t,
    _size_any_t,
)
from oneflow.compatible.single_client.nn.module import Module
from oneflow.compatible.single_client.nn.modules.utils import (
    _pair,
    _reverse_repeat_tuple,
    _single,
    _triple,
)


def mirrored_gen_random_seed(seed=None):
    if seed is None:
        seed = -1
        has_seed = False
    else:
        has_seed = True
    return (seed, has_seed)


class TensorBufferToListOfTensors(Module):
    def __init__(
        self, out_shapes, out_dtypes, out_num: int = 1, dynamic_out: bool = False
    ):
        super().__init__()
        self._op = (
            flow.builtin_op("tensor_buffer_to_list_of_tensors_v2")
            .Input("in")
            .Output("out", out_num)
            .Attr("out_shapes", out_shapes)
            .Attr("out_dtypes", out_dtypes)
            .Attr("dynamic_out", dynamic_out)
            .Build()
        )

    def forward(self, input):
        return self._op(input)
