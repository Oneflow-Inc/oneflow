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
import oneflow

# oneflow._C.max_poolXd returns a TensorTuple, to align torch,
# here we return different result according to the param `return_indices`.
def max_pool1d(
    x,
    kernel_size,
    stride=None,
    padding=0,
    dilation=1,
    return_indices=False,
    ceil_mode=False,
    data_format="channels_first",
):
    _max_pool_out = oneflow._C.max_pool1d(
        x,
        kernel_size,
        stride,
        padding,
        dilation,
        return_indices,
        ceil_mode,
        data_format,
    )
    if return_indices:
        return _max_pool_out
    else:
        return _max_pool_out[0]


def max_pool2d(
    x,
    kernel_size,
    stride=None,
    padding=0,
    dilation=1,
    return_indices=False,
    ceil_mode=False,
    data_format="channels_first",
):
    _max_pool_out = oneflow._C.max_pool2d(
        x,
        kernel_size,
        stride,
        padding,
        dilation,
        return_indices,
        ceil_mode,
        data_format,
    )
    if return_indices:
        return _max_pool_out
    else:
        return _max_pool_out[0]


def max_pool3d(
    x,
    kernel_size,
    stride=None,
    padding=0,
    dilation=1,
    return_indices=False,
    ceil_mode=False,
    data_format="channels_first",
):
    _max_pool_out = oneflow._C.max_pool3d(
        x,
        kernel_size,
        stride,
        padding,
        dilation,
        return_indices,
        ceil_mode,
        data_format,
    )
    if return_indices:
        return _max_pool_out
    else:
        return _max_pool_out[0]
