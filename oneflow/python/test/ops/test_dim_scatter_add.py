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
import numpy as np
import oneflow.typing as oft
from test_util import GenArgList
import unittest
from collections import OrderedDict
import os


def gen_scatter_add_like_test_sample(
    input_shape, index_shape, dim, like_shape, is_float=True
):
    def _np_dim_scatter_add_like(input, dim, index, like):
        output = np.zeros(like.shape)
        for inputidx in range(0, input.size):
            outcoord = np.unravel_index(inputidx, input.shape)
            outcoord = [*outcoord]
            outcoord[dim] = index[np.unravel_index(inputidx, index.shape)]
            output_offset = np.ravel_multi_index(outcoord, like_shape)
            output[np.unravel_index(output_offset, like_shape)] += input[
                np.unravel_index(inputidx, input.shape)
            ]
        return output
    like = np.random.randint(0, 100, like_shape)
    if is_float:
        input = np.random.random(input_shape)
    else:
        input = np.random.randint(0, 100, input_shape)

    def _np_dim_gather(dim, input, index):
        output = np.zeros(index.shape)
        for idx in range(0, index.size):
            incoord = np.unravel_index(idx, index.shape)
            outcoord=[*incoord]
            incoord = [*incoord]
            incoord[dim] = index[np.unravel_index(idx, index.shape)]
            output[tuple(outcoord)] = input[tuple(incoord)]
        return output

    index = np.random.randint(0, like_shape[dim], index_shape)

    output = _np_dim_scatter_add_like(input, dim, index, like)
    grad = _np_dim_gather(dim, output, index)
    return {
        "input": input,
        "index": index,
        "like": like,
        "dim": dim,
        "output": output,
        "grad": grad
    }

sample = gen_scatter_add_like_test_sample((2, 2), (2, 2), 0, (4,4))
print(sample)
