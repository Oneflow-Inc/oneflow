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

import os

if os.getenv("CTEST_RESOURCE_GROUP_COUNT"):
    vram_str = os.getenv("CTEST_RESOURCE_GROUP_0_VRAM")
    gpu_id = vram_str.split(",")[0].split(":")[-1]
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

import sys
import collections

import oneflow._oneflow_internal

oneflow._oneflow_internal.InitNumpyCAPI()
oneflow._oneflow_internal.CheckAndClearRegistryFlag()
Size = oneflow._oneflow_internal.Size
device = oneflow._oneflow_internal.device
placement = oneflow._oneflow_internal.placement
locals()["dtype"] = oneflow._oneflow_internal.dtype
locals()["bool"] = oneflow._oneflow_internal.bool
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
locals()["bfloat16"] = oneflow._oneflow_internal.bfloat16
from oneflow.version import __version__
from oneflow.version import __git_commit__

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


from oneflow._C import abs
from oneflow._C import exp
from oneflow._C import acos
from oneflow._C import acos as arccos
from oneflow._C import acosh
from oneflow._C import acosh as arccosh
from oneflow._C import atanh
from oneflow._C import atanh as arctanh
from oneflow._C import batch_matmul as bmm
from oneflow._C import broadcast_like
from oneflow._C import chunk
from oneflow._C import split
from oneflow._C import sign
from oneflow._C import sinh
from oneflow._C import tan
from oneflow._C import greater
from oneflow._C import greater as gt
from oneflow._C import greater_equal
from oneflow._C import greater_equal as ge
from oneflow._C import logical_and
from oneflow._C import logical_or
from oneflow._C import logical_xor
from oneflow._C import logical_not
from oneflow._C import gelu
from oneflow._C import mish
from oneflow._C import sigmoid
from oneflow._C import tanh
from oneflow._C import silu
from oneflow._C import selu
from oneflow._C import softsign
from oneflow._C import cast
from oneflow._C import ones_like
from oneflow._C import zeros_like
from oneflow._C import diag
from oneflow._C import log1p
from oneflow._C import add
from oneflow._C import div
from oneflow._C import floor
from oneflow._C import floor_divide
from oneflow._C import mul
from oneflow._C import negative
from oneflow._C import negative as neg
from oneflow._C import reciprocal
from oneflow._C import sub
from oneflow._C import sin, sin_
from oneflow._C import asin
from oneflow._C import asin as arcsin
from oneflow._C import asinh
from oneflow._C import asinh as arcsinh
from oneflow._C import atan
from oneflow._C import atan as arctan
from oneflow._C import atan2
from oneflow._C import ceil
from oneflow._C import clamp
from oneflow._C import clamp as clip
from oneflow._C import cos
from oneflow._C import cosh
from oneflow._C import diagonal
from oneflow._C import erf
from oneflow._C import erfc
from oneflow._C import expm1
from oneflow._C import fmod
from oneflow._C import flatten
from oneflow._C import log
from oneflow._C import log2
from oneflow._C import minimum
from oneflow._C import maximum
from oneflow._C import pow
from oneflow._C import rsqrt
from oneflow._C import sqrt
from oneflow._C import square
from oneflow._C import matmul
from oneflow._C import bernoulli
from oneflow._C import round
from oneflow._C import softplus
from oneflow._C import tril
from oneflow._C import triu
from oneflow._C import pad
from oneflow._C import distributed_partial_fc_sample
from oneflow._C import transpose
from oneflow._C import relu
from oneflow._C import softmax
from oneflow._C import log_softmax
from oneflow._C import argmax
from oneflow._C import argmin
from oneflow._C import std
from oneflow._C import var
from oneflow._C import meshgrid
from oneflow._C import stack
from oneflow._C import squeeze
from oneflow._C import narrow
from oneflow._C import unsqueeze
from oneflow._C import permute
from oneflow._C import concat
from oneflow._C import concat as cat
from oneflow._C import to
from oneflow._C import dim_gather as gather
from oneflow._C import gather_nd
from oneflow._C import roi_align
from oneflow._C import read_onerec
from oneflow._C import decode_onerec
from oneflow._C import dot
from oneflow._C import eye


from . import sbp
import atexit

import oneflow.framework.c_api_util
import oneflow.framework.register_class_method_util as register_class_method_util
import oneflow.framework.register_python_callback


INVALID_SPLIT_AXIS = oneflow._oneflow_internal.INVALID_SPLIT_AXIS
register_class_method_util.RegisterMethod4Class()
import oneflow.framework.env_util as env_util
import oneflow.framework.scope_util as scope_util
import oneflow.framework.session_context as session_ctx
from oneflow.framework.multi_client_session import MultiClientSession
from oneflow.framework.tensor_str import set_printoptions

if not env_util.HasAllMultiClientEnvVars():
    env_util.SetDefaultMultiClientEnvVars()
oneflow._oneflow_internal.SetIsMultiClient(True)
env_util.api_env_init()
oneflow._oneflow_internal.RegisterGILForeignLockHelper()
oneflow._oneflow_internal.InitDefaultConsistentTransportTokenScope()
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


class ExitHook:
    def __init__(self):
        self.exit_code = None
        self.exception = None

        self._orig_exit = sys.exit
        self._orig_excepthook = sys.excepthook

        def exit(code=0):
            self.exit_code = code
            self._orig_exit(code)

        sys.exit = exit

        def exc_handler(exc_type, exc, *args):
            self.exception = exc
            self._orig_excepthook(exc_type, exc, *args)

        sys.excepthook = exc_handler

    def is_normal_exit(self):
        if self.exit_code is not None:
            return self.exit_code == 0
        return self.exception is None


hook = ExitHook()


def atexit_hook(hook):
    if hook.is_normal_exit():
        if oneflow._oneflow_internal.IsEnvInited():
            if oneflow.env.is_multi_client():
                oneflow._oneflow_internal.eager.multi_client.Sync()
            elif oneflow.env.get_rank() == 0:
                oneflow._oneflow_internal.eager.single_client.Sync()
    oneflow.framework.session_context.TryCloseDefaultSession()
    if hook.is_normal_exit():
        oneflow._oneflow_internal.DestroyEnv()
    oneflow._oneflow_internal.SetShuttingDown()


atexit.register(atexit_hook, hook)
del atexit_hook
del hook
del ExitHook
del atexit
del oneflow

import oneflow._C
from oneflow._C import tensor, batch_gather
from oneflow._C import from_numpy

from oneflow.autograd import grad_enable, no_grad, inference_mode, is_grad_enabled
import oneflow.nn.image

from oneflow.framework.check_point_v2 import load
from oneflow.framework.check_point_v2 import save
from oneflow.framework.dtype import convert_oneflow_dtype_to_numpy_dtype, dtypes
from oneflow.framework.env_util import (
    api_enable_eager_execution as enable_eager_execution,
)
from oneflow.framework.function_util import FunctionConfig
from oneflow.framework.function_util import FunctionConfig as function_config
from oneflow.framework.generator import create_generator as Generator
from oneflow.framework.generator import (
    default_generator,
    manual_seed,
    get_rng_state,
    set_rng_state,
)

# NOTE(chengcheng) oneflow.Model is unavailable now.
# from oneflow.framework.model import Model
from oneflow.framework.scope_util import api_current_scope as current_scope
from oneflow.framework.tensor import Tensor
from oneflow.framework.tensor import is_nonzero

from oneflow.nn.modules.pooling import (
    adaptive_avg_pool1d,
    adaptive_avg_pool2d,
    adaptive_avg_pool3d,
)
from oneflow.nn.modules.arange import arange_op as arange
from oneflow.nn.modules.linspace import linspace_op as linspace
from oneflow.nn.modules.argsort import argsort_op as argsort
from oneflow.nn.modules.argwhere import argwhere_op as argwhere
from oneflow.nn.modules.constant import ones_op as ones
from oneflow.nn.modules.constant import zeros_op as zeros
from oneflow.nn.modules.constant import full_op as full
from oneflow.nn.modules.empty import empty_op as empty
from oneflow.nn.modules.dataset import tensor_buffer_to_list_of_tensors
from oneflow._C import movedim
from oneflow.nn.modules.expand import expand_op as expand
from oneflow.nn.modules.roll import roll_op as roll
from oneflow.nn.modules.flip import flip_op as flip
from oneflow.nn.modules.comparison import eq_op as eq
from oneflow.nn.modules.comparison import eq_op as equal
from oneflow.nn.modules.logical_ops import logical_and_op as logical_and
from oneflow.nn.modules.logical_ops import logical_or_op as logical_or
from oneflow.nn.modules.logical_ops import logical_xor_op as logical_xor
from oneflow.nn.modules.comparison import less_op as lt
from oneflow.nn.modules.comparison import less_equal_op as le
from oneflow.nn.modules.comparison import ne_op as ne
from oneflow.nn.modules.comparison import ne_op as not_equal
from oneflow.nn.modules.tensor_ops import is_floating_point
from oneflow.nn.modules.in_top_k import in_top_k_op as in_top_k
from oneflow.nn.modules.index_select import index_select_op as index_select
from oneflow.nn.modules.masked_fill import masked_fill_op as masked_fill
from oneflow.nn.modules.masked_select import masked_select_op as masked_select
from oneflow.nn.modules.math_ops import addmm_op as addmm
from oneflow.nn.modules.math_ops import topk_op as topk
from oneflow.nn.modules.nonzero import nonzero_op as nonzero
from oneflow.nn.modules.nms import nms_op as nms
from oneflow.nn.modules.numel import numel_op as numel
from oneflow.nn.modules.random_ops import rand_op as rand
from oneflow.nn.modules.random_ops import randn_op as randn
from oneflow.nn.modules.random_ops import randint_op as randint
from oneflow.nn.modules.random_ops import randperm_op as randperm
from oneflow.nn.modules.reduce_ops import max_op as max
from oneflow.nn.modules.reduce_ops import min_op as min
from oneflow.nn.modules.reduce_ops import sum_op as sum
from oneflow.nn.modules.reduce_ops import mean_op as mean
from oneflow.nn.modules.reduce_ops import prod_op as prod
from oneflow.nn.modules.reduce_ops import all_op as all
from oneflow.nn.modules.reduce_ops import any_op as any
from oneflow.nn.modules.repeat import repeat_op as repeat
from oneflow.nn.modules.reshape import reshape_op as reshape
from oneflow.nn.modules.reshape import view_op as view
from oneflow.nn.modules.slice import slice_op as slice
from oneflow.nn.modules.slice import slice_update_op as slice_update
from oneflow.nn.modules.slice import logical_slice_assign_op as logical_slice_assign
from oneflow.nn.modules.sort import sort_op as sort
from oneflow.nn.modules.tensor_buffer import gen_tensor_buffer
from oneflow.nn.modules.tensor_buffer import (
    tensor_buffer_to_tensor_op as tensor_buffer_to_tensor,
)
from oneflow.nn.modules.as_tensor import as_tensor
from oneflow.nn.modules.tensor_buffer import tensor_to_tensor_buffer
from oneflow.nn.modules.tile import tile_op as tile
from oneflow.nn.modules.consistent_cast import to_consistent_op as to_consistent
from oneflow.nn.modules.consistent_cast import to_local_op as to_local
from oneflow.nn.modules.where import where_op as where
from oneflow.nn.modules.scatter import *
from oneflow.ops.builtin_ops import BuiltinOp as builtin_op
from oneflow.ops.initializer_util import constant_initializer
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


from . import (
    autograd,
    distributed,
    linalg,
    optim,
    comm,
    boxing,
    backends,
    amp,
)  # , saved_model NOTE(chengcheng): unavailable now
import oneflow.utils.data
import oneflow.utils.vision
import oneflow.comm
import oneflow.framework.docstr as docstr
import oneflow.cuda
import oneflow.multiprocessing

if oneflow._oneflow_internal.flags.with_mlir():
    oneflow_internal_path = oneflow._oneflow_internal.__file__
    if os.getenv("ONEFLOW_MLIR_ENABLE_CODEGEN_FUSERS"):
        print("MLIR JIT engine will load:", oneflow_internal_path, file=sys.stderr)
        oneflow._oneflow_internal.ir.load_jit_shared_lib(oneflow_internal_path)
