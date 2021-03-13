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

from oneflow.python.oneflow_export import oneflow_export
from oneflow.python.nn.module import Module
from oneflow.python.nn.modules.utils import (
    _single,
    _pair,
    _triple,
    _reverse_repeat_tuple,
)
from oneflow.python.nn.common_types import _size_1_t, _size_2_t, _size_3_t
from typing import Optional, List, Tuple
from oneflow.python.ops.nn_ops import calc_pool_padding, get_dhw_offset
import oneflow.python.framework.id_util as id_util


@oneflow_export("nn.AvgPool2d")
class AvgPool2d(Module):
    r"""Applies a 2D average pooling over an input signal composed of several input
    planes.
    """

    def __init__(
        self,
        kernel_size: _size_2_t,
        stride: Optional[_size_2_t] = None,
        padding: _size_2_t = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = None,
        divisor_override: Optional[int] = None,
    ):
        super().__init__()
        kernel_size = _pair(kernel_size)
        stride = _pair(stride) if (stride is not None) else kernel_size
        if padding == 0:
            padding = "VALID"
        else:
            raise ValueError("padding != 0 not supported yet")

        ceil_mode = ceil_mode

        assert count_include_pad is None, "count_include_pad not supported yet"
        assert divisor_override is None, "divisor_override not supported yet"
        count_include_pad = count_include_pad
        divisor_override = divisor_override

        _opname = id_util.UniqueStr("Module_AvgPool2D_")
        _channel_pos = "channels_first"

        _padding_type, _pads_list = calc_pool_padding(
            padding, get_dhw_offset(_channel_pos), 2
        )
        _padding_before = [pad[0] for pad in _pads_list]
        _padding_after = [pad[1] for pad in _pads_list]

        self._op = (
            flow.builtin_op("avg_pool_2d")
            .Name(_opname)
            .Attr("data_format", _channel_pos)
            .Attr("pool_size", kernel_size)
            .Attr("strides", stride)
            .Attr("ceil_mode", ceil_mode)
            .Attr("padding", _padding_type)
            .Attr("padding_before", _padding_before)
            .Attr("padding_after", _padding_after)
            .Input("x")
            .Output("y")
            .Build()
        )

    def forward(self, x):
        res = self._op(x)[0]
        return res
