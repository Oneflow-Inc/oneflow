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

from oneflow.compatible.single_client import unittest
from oneflow.compatible.single_client.experimental.indexed_slices_ops import (
    indexed_slices_reduce_sum,
)
from oneflow.compatible.single_client.experimental.interface_op_read_and_write import (
    FeedValueToInterfaceBlob as set_interface_blob_value,
)
from oneflow.compatible.single_client.experimental.interface_op_read_and_write import (
    GetInterfaceBlobValue as get_interface_blob_value,
)
from oneflow.compatible.single_client.experimental.namescope import (
    deprecated_name_scope as name_scope,
)
from oneflow.compatible.single_client.experimental.square_sum_op import square_sum
from oneflow.compatible.single_client.experimental.ssp_variable_proxy_op import (
    ssp_variable_proxy,
)
from oneflow.compatible.single_client.experimental.typing_check import (
    api_enable_typing_check as enable_typing_check,
)
from oneflow.compatible.single_client.experimental.unique_op import unique_with_counts
from oneflow.compatible.single_client.framework.c_api_util import (
    GetJobSet as get_job_set,
)
from oneflow.compatible.single_client.nn.modules.abs import abs_op as abs
from oneflow.compatible.single_client.nn.modules.acos import acos_op as acos
from oneflow.compatible.single_client.nn.modules.acosh import acosh_op as acosh
from oneflow.compatible.single_client.nn.modules.acosh import arccosh_op as arccosh
from oneflow.compatible.single_client.nn.modules.activation import gelu_op as gelu
from oneflow.compatible.single_client.nn.modules.activation import mish_op as mish
from oneflow.compatible.single_client.nn.modules.activation import sigmoid_op as sigmoid
from oneflow.compatible.single_client.nn.modules.activation import softmax_op as softmax
from oneflow.compatible.single_client.nn.modules.activation import tanh_op as tanh
from oneflow.compatible.single_client.nn.modules.arange import arange_op as arange
from oneflow.compatible.single_client.nn.modules.argmax import argmax_op as argmax
from oneflow.compatible.single_client.nn.modules.argsort import argsort_op as argsort
from oneflow.compatible.single_client.nn.modules.argwhere import argwhere_op as argwhere
from oneflow.compatible.single_client.nn.modules.atan2 import atan2_op as atan2
from oneflow.compatible.single_client.nn.modules.atanh import arctanh_op as arctanh
from oneflow.compatible.single_client.nn.modules.atanh import atanh_op as atanh
from oneflow.compatible.single_client.nn.modules.bmm import bmm_op as bmm
from oneflow.compatible.single_client.nn.modules.broadcast_like import (
    broadcast_like_op as broadcast_like,
)
from oneflow.compatible.single_client.nn.modules.cast import cast_op as cast
from oneflow.compatible.single_client.nn.modules.chunk import chunk_op as chunk
from oneflow.compatible.single_client.nn.modules.concat import concat_op as cat
from oneflow.compatible.single_client.nn.modules.constant import (
    ones_like_op as ones_like,
)
from oneflow.compatible.single_client.nn.modules.constant import ones_op as ones
from oneflow.compatible.single_client.nn.modules.constant import (
    zeros_like_op as zeros_like,
)
from oneflow.compatible.single_client.nn.modules.constant import zeros_op as zeros
from oneflow.compatible.single_client.nn.modules.dataset import (
    tensor_buffer_to_list_of_tensors,
)
from oneflow.compatible.single_client.nn.modules.eq import eq_op as eq
from oneflow.compatible.single_client.nn.modules.eq import eq_op as equal
from oneflow.compatible.single_client.nn.modules.exp import exp_op as exp
from oneflow.compatible.single_client.nn.modules.expand import expand_op as expand
from oneflow.compatible.single_client.nn.modules.flatten import _flow_flatten as flatten
from oneflow.compatible.single_client.nn.modules.floor import floor_op as floor
from oneflow.compatible.single_client.nn.modules.gather import gather_op as gather
from oneflow.compatible.single_client.nn.modules.greater import greater_op as gt
from oneflow.compatible.single_client.nn.modules.greater_equal import (
    greater_equal_op as ge,
)
from oneflow.compatible.single_client.nn.modules.less import less_op as lt
from oneflow.compatible.single_client.nn.modules.less_equal import less_equal_op as le
from oneflow.compatible.single_client.nn.modules.log1p import log1p_op as log1p
from oneflow.compatible.single_client.nn.modules.masked_fill import (
    masked_fill_op as masked_fill,
)
from oneflow.compatible.single_client.nn.modules.masked_select import (
    masked_select_op as masked_select,
)
from oneflow.compatible.single_client.nn.modules.math_ops import _add as add
from oneflow.compatible.single_client.nn.modules.math_ops import _div as div
from oneflow.compatible.single_client.nn.modules.math_ops import _mul as mul
from oneflow.compatible.single_client.nn.modules.math_ops import (
    _reciprocal as reciprocal,
)
from oneflow.compatible.single_client.nn.modules.math_ops import _sub as sub
from oneflow.compatible.single_client.nn.modules.math_ops import addmm_op as addmm
from oneflow.compatible.single_client.nn.modules.math_ops import arcsin_op as arcsin
from oneflow.compatible.single_client.nn.modules.math_ops import arcsinh_op as arcsinh
from oneflow.compatible.single_client.nn.modules.math_ops import arctan_op as arctan
from oneflow.compatible.single_client.nn.modules.math_ops import asin_op as asin
from oneflow.compatible.single_client.nn.modules.math_ops import asinh_op as asinh
from oneflow.compatible.single_client.nn.modules.math_ops import atan_op as atan
from oneflow.compatible.single_client.nn.modules.math_ops import ceil_op as ceil
from oneflow.compatible.single_client.nn.modules.math_ops import clamp_op as clamp
from oneflow.compatible.single_client.nn.modules.math_ops import clip_op as clip
from oneflow.compatible.single_client.nn.modules.math_ops import cos_op as cos
from oneflow.compatible.single_client.nn.modules.math_ops import cosh_op as cosh
from oneflow.compatible.single_client.nn.modules.math_ops import erf_op as erf
from oneflow.compatible.single_client.nn.modules.math_ops import erfc_op as erfc
from oneflow.compatible.single_client.nn.modules.math_ops import expm1_op as expm1
from oneflow.compatible.single_client.nn.modules.math_ops import log_op as log
from oneflow.compatible.single_client.nn.modules.math_ops import pow_op as pow
from oneflow.compatible.single_client.nn.modules.math_ops import rsqrt_op as rsqrt
from oneflow.compatible.single_client.nn.modules.math_ops import sin_op as sin
from oneflow.compatible.single_client.nn.modules.math_ops import sqrt_op as sqrt
from oneflow.compatible.single_client.nn.modules.math_ops import square_op as square
from oneflow.compatible.single_client.nn.modules.math_ops import std_op as std
from oneflow.compatible.single_client.nn.modules.math_ops import topk_op as topk
from oneflow.compatible.single_client.nn.modules.math_ops import variance_op as var
from oneflow.compatible.single_client.nn.modules.matmul import matmul_op as matmul
from oneflow.compatible.single_client.nn.modules.meshgrid import meshgrid_op as meshgrid
from oneflow.compatible.single_client.nn.modules.ne import ne_op as ne
from oneflow.compatible.single_client.nn.modules.ne import ne_op as not_equal
from oneflow.compatible.single_client.nn.modules.negative import negative_op as neg
from oneflow.compatible.single_client.nn.modules.negative import negative_op as negative
from oneflow.compatible.single_client.nn.modules.reduce_ops import _max as max
from oneflow.compatible.single_client.nn.modules.reduce_ops import _mean as mean
from oneflow.compatible.single_client.nn.modules.reduce_ops import _min as min
from oneflow.compatible.single_client.nn.modules.reduce_ops import _sum as sum
from oneflow.compatible.single_client.nn.modules.repeat import repeat_op as repeat
from oneflow.compatible.single_client.nn.modules.reshape import reshape_op as reshape
from oneflow.compatible.single_client.nn.modules.reshape import view_op as view
from oneflow.compatible.single_client.nn.modules.round import round_op as round
from oneflow.compatible.single_client.nn.modules.sign import sign_op as sign
from oneflow.compatible.single_client.nn.modules.sinh import sinh_op as sinh
from oneflow.compatible.single_client.nn.modules.slice import slice_op as slice
from oneflow.compatible.single_client.nn.modules.slice import (
    slice_update_op as slice_update,
)
from oneflow.compatible.single_client.nn.modules.softplus import softplus_op as softplus
from oneflow.compatible.single_client.nn.modules.sort import sort_op as sort
from oneflow.compatible.single_client.nn.modules.squeeze import squeeze_op as squeeze
from oneflow.compatible.single_client.nn.modules.stack import stack
from oneflow.compatible.single_client.nn.modules.tan import tan_op as tan
from oneflow.compatible.single_client.nn.modules.tensor_buffer import gen_tensor_buffer
from oneflow.compatible.single_client.nn.modules.tensor_buffer import (
    tensor_buffer_to_tensor_op as tensor_buffer_to_tensor,
)
from oneflow.compatible.single_client.nn.modules.tensor_buffer import (
    tensor_to_tensor_buffer,
)
from oneflow.compatible.single_client.nn.modules.tile import tile_op as tile
from oneflow.compatible.single_client.nn.modules.transpose import (
    transpose_op as transpose,
)
from oneflow.compatible.single_client.nn.modules.triu import triu_op as triu
from oneflow.compatible.single_client.nn.modules.unsqueeze import (
    unsqueeze_op as unsqueeze,
)
from oneflow.compatible.single_client.nn.modules.where import where_op as where
from oneflow.compatible.single_client.ops.array_ops import (
    logical_slice,
    logical_slice_assign,
)
from oneflow.compatible.single_client.ops.assign_op import (
    api_one_to_one_assign as eager_assign_121,
)
from oneflow.compatible.single_client.ops.util.custom_op_module import (
    CustomOpModule as custom_op_module,
)

from . import scope
