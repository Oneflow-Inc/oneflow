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

import collections

import oneflow._oneflow_internal

oneflow._oneflow_internal.CheckAndClearRegistryFlag()
Size = oneflow._oneflow_internal.Size
device = oneflow._oneflow_internal.device
placement = oneflow._oneflow_internal.placement
no_grad = oneflow._oneflow_internal.autograd.no_grad
locals()["dtype"] = oneflow._oneflow_internal.dtype
locals()["char"] = oneflow._oneflow_internal.char
locals()["float16"] = oneflow._oneflow_internal.float16
locals()["half"] = oneflow._oneflow_internal.float16
locals()["float32"] = oneflow._oneflow_internal.float32
locals()["float"] = oneflow._oneflow_internal.float
locals()["double"] = oneflow._oneflow_internal.double
locals()["float64"] = oneflow._oneflow_internal.float64
locals()["int8"] = oneflow._oneflow_internal.int8
locals()["int"] = oneflow._oneflow_internal.int32
locals()["int32"] = oneflow._oneflow_internal.int32
locals()["int64"] = oneflow._oneflow_internal.int64
locals()["long"] = oneflow._oneflow_internal.int64
locals()["uint8"] = oneflow._oneflow_internal.uint8
locals()["record"] = oneflow._oneflow_internal.record
locals()["tensor_buffer"] = oneflow._oneflow_internal.tensor_buffer
from oneflow.core.job.job_conf_pb2 import JobConfigProto
from oneflow.core.job.job_set_pb2 import ConfigProto
from oneflow.version import __version__

_DEPRECATED = set()


def oneflow_deprecate(*api_names, **kwargs):
    def Decorator(func_or_class):
        _DEPRECATED.add(func_or_class)
        return func_or_class

    return Decorator


def is_deprecated(func_or_class):
    return (
        isinstance(func_or_class, collections.Hashable) and func_or_class in _DEPRECATED
    )


import atexit

import oneflow.framework.c_api_util
import oneflow.framework.register_class_method_util as register_class_method_util
import oneflow.framework.register_python_callback

INVALID_SPLIT_AXIS = oneflow._oneflow_internal.INVALID_SPLIT_AXIS
register_class_method_util.RegisterMethod4Class()
oneflow._oneflow_internal.RegisterGILForeignLockHelper()
import oneflow.framework.env_util as env_util
import oneflow.framework.scope_util as scope_util
import oneflow.framework.session_context as session_ctx
from oneflow.framework.multi_client_session import MultiClientSession
from oneflow.framework.session_util import Session

if not env_util.HasAllMultiClientEnvVars():
    env_util.SetDefaultMultiClientEnvVars()
oneflow._oneflow_internal.SetIsMultiClient(True)
env_util.api_env_init()
oneflow._oneflow_internal.InitDefaultConsistentRpcTokenScope()
session_ctx.OpenDefaultSession(
    MultiClientSession(oneflow._oneflow_internal.NewSessionId())
)
scope_util.InitScopeStack()
oneflow._oneflow_internal.EnableEagerEnvironment(True)
del env_util
from oneflow.framework import python_callback, register_python_callback

oneflow._oneflow_internal.RegisterGlobalForeignCallback(
    python_callback.global_python_callback
)
del python_callback
del register_python_callback
from oneflow.framework import watcher

oneflow._oneflow_internal.RegisterGlobalWatcher(watcher._global_watcher)
del watcher


def _SyncOnMasterFn():
    import oneflow

    def Sync():
        if not oneflow._oneflow_internal.IsEnvInited():
            return
        if oneflow.framework.distribute.is_multi_client():
            oneflow._oneflow_internal.eager.multi_client.Sync()
        elif oneflow.framework.distribute.get_rank() == 0:
            oneflow._oneflow_internal.eager.single_client.Sync()

    return Sync


atexit.register(oneflow._oneflow_internal.SetShuttingDown)
atexit.register(oneflow._oneflow_internal.DestroyEnv)
atexit.register(oneflow.framework.session_context.TryCloseDefaultSession)
atexit.register(_SyncOnMasterFn)
del atexit
del oneflow
import oneflow.framework.docstr as docstr
from oneflow.framework.docstr.utils import register_docstr

register_docstr()
del register_docstr
del docstr
import oneflow.nn.image
import oneflow.nn.modules.acosh
import oneflow.nn.modules.activation
import oneflow.nn.modules.argwhere
import oneflow.nn.modules.atan2
import oneflow.nn.modules.atanh
import oneflow.nn.modules.bmm
import oneflow.nn.modules.constant
import oneflow.nn.modules.diag
import oneflow.nn.modules.flip
import oneflow.nn.modules.floor
import oneflow.nn.modules.greater
import oneflow.nn.modules.greater_equal
import oneflow.nn.modules.in_top_k
import oneflow.nn.modules.masked_select
import oneflow.nn.modules.math_ops
import oneflow.nn.modules.norm
import oneflow.nn.modules.permute
import oneflow.nn.modules.round
import oneflow.nn.modules.sign
import oneflow.nn.modules.sinh
import oneflow.nn.modules.tan
import oneflow.nn.modules.tensor_ops
import oneflow.tmp
from oneflow.advanced.distribute_ops import cast_to_current_logical_view
from oneflow.deprecated.initializer_util import (
    truncated_normal_initializer as truncated_normal,
)
from oneflow.experimental.namescope import deprecated_name_scope as name_scope
from oneflow.framework.check_point_v2 import GetAllVariables as get_all_variables
from oneflow.framework.check_point_v2 import Load as load
from oneflow.framework.check_point_v2 import LoadVariables as load_variables
from oneflow.framework.check_point_v2 import save
from oneflow.framework.dtype import convert_oneflow_dtype_to_numpy_dtype, dtypes
from oneflow.framework.env_util import (
    api_enable_eager_execution as enable_eager_execution,
)
from oneflow.framework.env_util import api_get_current_machine_id as current_machine_id
from oneflow.framework.env_util import api_get_current_resource as current_resource
from oneflow.framework.function_desc import (
    api_current_global_function_desc as current_global_function_desc,
)
from oneflow.framework.function_util import FunctionConfig
from oneflow.framework.function_util import FunctionConfig as ExecutionConfig
from oneflow.framework.function_util import FunctionConfig as function_config
from oneflow.framework.function_util import api_oneflow_function as global_function
from oneflow.framework.generator import create_generator as Generator
from oneflow.framework.generator import default_generator, manual_seed
from oneflow.framework.input_blob_def import DeprecatedFixedTensorDef as FixedTensorDef
from oneflow.framework.input_blob_def import (
    DeprecatedMirroredTensorDef as MirroredTensorDef,
)
from oneflow.framework.job_set_util import inter_job_reuse_mem_strategy
from oneflow.framework.model import Model
from oneflow.framework.ops import api_acc as acc
from oneflow.framework.ops import (
    api_hierarchical_parallel_cast as hierarchical_parallel_cast,
)
from oneflow.framework.ops import api_pack as pack
from oneflow.framework.ops import api_parallel_cast as parallel_cast
from oneflow.framework.ops import api_unpack as unpack
from oneflow.framework.placement_util import (
    deprecated_placement as device_prior_placement,
)
from oneflow.framework.placement_util import deprecated_placement as fixed_placement
from oneflow.framework.scope_util import api_current_scope as current_scope
from oneflow.framework.session_util import (
    TmpInitEagerGlobalSession as InitEagerGlobalSession,
)
from oneflow.framework.session_util import (
    api_clear_default_session as clear_default_session,
)
from oneflow.framework.session_util import (
    api_eager_execution_enabled as eager_execution_enabled,
)
from oneflow.framework.session_util import (
    api_find_or_create_module as find_or_create_module,
)
from oneflow.framework.session_util import (
    api_sync_default_session as sync_default_session,
)
from oneflow.framework.tensor import Tensor
from oneflow.framework.tensor import tensor as tensor
from oneflow.nn.modules.abs import abs_op as abs
from oneflow.nn.modules.acos import acos_op as acos
from oneflow.nn.modules.acosh import acosh_op as acosh
from oneflow.nn.modules.acosh import arccosh_op as arccosh
from oneflow.nn.modules.activation import gelu_op as gelu
from oneflow.nn.modules.activation import mish_op as mish
from oneflow.nn.modules.activation import sigmoid_op as sigmoid
from oneflow.nn.modules.activation import softmax_op as softmax
from oneflow.nn.modules.activation import tanh_op as tanh
from oneflow.nn.modules.activation import silu_op as silu
from oneflow.nn.modules.activation import selu_op as selu
from oneflow.nn.modules.activation import softsign_op as softsign
from oneflow.nn.modules.activation import mish_op as mish


from oneflow.nn.modules.adaptive_pool import (
    adaptive_avg_pool1d,
    adaptive_avg_pool2d,
    adaptive_avg_pool3d,
)
from oneflow.nn.modules.arange import arange_op as arange
from oneflow.nn.modules.argmax import argmax_op as argmax
from oneflow.nn.modules.argsort import argsort_op as argsort
from oneflow.nn.modules.argwhere import argwhere_op as argwhere
from oneflow.nn.modules.atan2 import atan2_op as atan2
from oneflow.nn.modules.atanh import arctanh_op as arctanh
from oneflow.nn.modules.atanh import atanh_op as atanh
from oneflow.nn.modules.bmm import bmm_op as bmm
from oneflow.nn.modules.broadcast_like import broadcast_like_op as broadcast_like
from oneflow.nn.modules.cast import cast_op as cast
from oneflow.nn.modules.chunk import chunk_op as chunk
from oneflow.nn.modules.concat import concat_op as cat
from oneflow.nn.modules.constant import ones_like_op as ones_like
from oneflow.nn.modules.constant import ones_op as ones
from oneflow.nn.modules.constant import zeros_like_op as zeros_like
from oneflow.nn.modules.constant import zeros_op as zeros
from oneflow.nn.modules.dataset import tensor_buffer_to_list_of_tensors
from oneflow.nn.modules.diag import diag_op as diag
from oneflow.nn.modules.eq import eq_op as eq
from oneflow.nn.modules.eq import eq_op as equal
from oneflow.nn.modules.exp import exp_op as exp
from oneflow.nn.modules.expand import expand_op as expand
from oneflow.nn.modules.flatten import _flow_flatten as flatten
from oneflow.nn.modules.flip import flip_op as flip
from oneflow.nn.modules.floor import floor_op as floor
from oneflow.nn.modules.gather import gather_op as gather
from oneflow.nn.modules.gather_nd import gather_nd_op as gather_nd
from oneflow.nn.modules.greater import greater_op as gt
from oneflow.nn.modules.greater_equal import greater_equal_op as ge
from oneflow.nn.modules.in_top_k import in_top_k_op as in_top_k
from oneflow.nn.modules.less import less_op as lt
from oneflow.nn.modules.less_equal import less_equal_op as le
from oneflow.nn.modules.log1p import log1p_op as log1p
from oneflow.nn.modules.masked_fill import masked_fill_op as masked_fill
from oneflow.nn.modules.masked_select import masked_select_op as masked_select
from oneflow.nn.modules.math_ops import _add as add
from oneflow.nn.modules.math_ops import _div as div
from oneflow.nn.modules.math_ops import _mul as mul
from oneflow.nn.modules.math_ops import _reciprocal as reciprocal
from oneflow.nn.modules.math_ops import _sub as sub
from oneflow.nn.modules.math_ops import addmm_op as addmm
from oneflow.nn.modules.math_ops import arcsin_op as arcsin
from oneflow.nn.modules.math_ops import arcsinh_op as arcsinh
from oneflow.nn.modules.math_ops import arctan_op as arctan
from oneflow.nn.modules.math_ops import asin_op as asin
from oneflow.nn.modules.math_ops import asinh_op as asinh
from oneflow.nn.modules.math_ops import atan_op as atan
from oneflow.nn.modules.math_ops import ceil_op as ceil
from oneflow.nn.modules.math_ops import clamp_op as clamp
from oneflow.nn.modules.math_ops import clip_op as clip
from oneflow.nn.modules.math_ops import cos_op as cos
from oneflow.nn.modules.math_ops import cosh_op as cosh
from oneflow.nn.modules.math_ops import erf_op as erf
from oneflow.nn.modules.math_ops import erfc_op as erfc
from oneflow.nn.modules.math_ops import expm1_op as expm1
from oneflow.nn.modules.math_ops import fmod_op as fmod
from oneflow.nn.modules.math_ops import log_op as log
from oneflow.nn.modules.math_ops import minimum as minimum
from oneflow.nn.modules.math_ops import maximum as maximum
from oneflow.nn.modules.math_ops import pow_op as pow
from oneflow.nn.modules.math_ops import rsqrt_op as rsqrt
from oneflow.nn.modules.math_ops import sin_op as sin
from oneflow.nn.modules.math_ops import sqrt_op as sqrt
from oneflow.nn.modules.math_ops import square_op as square
from oneflow.nn.modules.math_ops import std_op as std
from oneflow.nn.modules.math_ops import topk_op as topk
from oneflow.nn.modules.math_ops import variance_op as var
from oneflow.nn.modules.matmul import matmul_op as matmul
from oneflow.nn.modules.meshgrid import meshgrid_op as meshgrid
from oneflow.nn.modules.ne import ne_op as ne
from oneflow.nn.modules.ne import ne_op as not_equal
from oneflow.nn.modules.negative import negative_op as neg
from oneflow.nn.modules.negative import negative_op as negative
from oneflow.nn.modules.random_ops import bernoulli
from oneflow.nn.modules.reduce_ops import _max as max
from oneflow.nn.modules.reduce_ops import _mean as mean
from oneflow.nn.modules.reduce_ops import _min as min
from oneflow.nn.modules.reduce_ops import _sum as sum
from oneflow.nn.modules.repeat import repeat_op as repeat
from oneflow.nn.modules.reshape import reshape_op as reshape
from oneflow.nn.modules.reshape import view_op as view
from oneflow.nn.modules.round import round_op as round
from oneflow.nn.modules.scatter_nd import _scatter_nd_op as scatter_nd
from oneflow.nn.modules.sign import sign_op as sign
from oneflow.nn.modules.sinh import sinh_op as sinh
from oneflow.nn.modules.slice import slice_op as slice
from oneflow.nn.modules.slice import slice_update_op as slice_update
from oneflow.nn.modules.softplus import softplus_op as softplus
from oneflow.nn.modules.sort import sort_op as sort
from oneflow.nn.modules.squeeze import squeeze_op as squeeze
from oneflow.nn.modules.stack import stack
from oneflow.nn.modules.tan import tan_op as tan
from oneflow.nn.modules.eye import eye_op as eye
from oneflow.nn.modules.tensor_buffer import gen_tensor_buffer
from oneflow.nn.modules.tensor_buffer import (
    tensor_buffer_to_tensor_op as tensor_buffer_to_tensor,
)
from oneflow.nn.modules.tensor_buffer import tensor_to_tensor_buffer
from oneflow.nn.modules.tile import tile_op as tile
from oneflow.nn.modules.to import to_op as to
from oneflow.nn.modules.transpose import transpose_op as transpose
from oneflow.nn.modules.triu import triu_op as triu
from oneflow.nn.modules.unsqueeze import unsqueeze_op as unsqueeze
from oneflow.nn.modules.where import where_op as where
from oneflow.ops.assign_op import assign
from oneflow.ops.builtin_ops import BuiltinOp as builtin_op
from oneflow.ops.categorical_ordinal_encode_op import categorical_ordinal_encode
from oneflow.ops.constant_op import constant, constant_like, constant_scalar
from oneflow.ops.count_not_finite import count_not_finite, multi_count_not_finite
from oneflow.ops.eager_nccl_ops import eager_nccl_all_reduce
from oneflow.ops.get_variable import api_get_variable as get_variable
from oneflow.ops.initializer_util import constant_initializer, empty_initializer
from oneflow.ops.initializer_util import glorot_normal_initializer
from oneflow.ops.initializer_util import (
    glorot_normal_initializer as xavier_normal_initializer,
)
from oneflow.ops.initializer_util import glorot_uniform_initializer
from oneflow.ops.initializer_util import (
    glorot_uniform_initializer as xavier_uniform_initializer,
)
from oneflow.ops.initializer_util import (
    kaiming_initializer,
    ones_initializer,
    random_normal_initializer,
    random_uniform_initializer,
    truncated_normal_initializer,
    variance_scaling_initializer,
    zeros_initializer,
)
from oneflow.ops.loss_ops import ctc_loss, smooth_l1_loss
from oneflow.ops.one_hot import one_hot
from oneflow.ops.partial_fc_sample import distributed_partial_fc_sample
from oneflow.ops.user_op_builder import (
    api_consistent_user_op_builder as consistent_user_op_builder,
)
from oneflow.ops.user_op_builder import (
    api_consistent_user_op_module_builder as consistent_user_op_module_builder,
)
from oneflow.ops.user_op_builder import api_user_op_builder as user_op_builder
from oneflow.ops.user_op_builder import (
    api_user_op_module_builder as user_op_module_builder,
)
from oneflow.nn.modules.scatter import *

from . import autograd, distributed, linalg, optim, saved_model, sbp
