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
import sys
import collections
import warnings

# https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-environment-variables
if "CUDA_MODULE_LOADING" not in os.environ:
    os.environ["CUDA_MODULE_LOADING"] = "LAZY"

import oneflow._oneflow_internal

oneflow_python_base_dir = os.path.dirname(os.path.realpath(__file__))
oneflow._oneflow_internal.InitPythonPathsToBeKeptAndFilteredForDebugging(
    oneflow_python_base_dir
)
oneflow._oneflow_internal.InitNumpyCAPI()
oneflow._oneflow_internal.CheckAndClearRegistryFlag()
Size = oneflow._oneflow_internal.Size
device = oneflow._oneflow_internal.device
placement = oneflow._oneflow_internal.placement

locals()["dtype"] = oneflow._oneflow_internal.dtype
locals()["bool"] = oneflow._oneflow_internal.bool
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

locals()["cfloat"] = oneflow._oneflow_internal.cfloat
locals()["complex64"] = oneflow._oneflow_internal.complex64
locals()["cdouble"] = oneflow._oneflow_internal.cdouble
locals()["complex128"] = oneflow._oneflow_internal.complex128

locals()["layout"] = oneflow._oneflow_internal.layout
locals()["strided"] = oneflow._oneflow_internal.strided

locals()["memory_format"] = oneflow._oneflow_internal.memory_format
locals()["contiguous_format"] = oneflow._oneflow_internal.contiguous_format
locals()["preserve_format"] = oneflow._oneflow_internal.preserve_format
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


def use_deterministic_algorithms(mode, *, warn_only=False):
    # register a empty method
    warnings.warn("Oneflow temporarily does not support use_deterministic_algorithms.")


from oneflow._C import abs
from oneflow._C import exp
from oneflow._C import exp2
from oneflow._C import acos
from oneflow._C import acos as arccos
from oneflow._C import acosh
from oneflow._C import acosh as arccosh
from oneflow._C import amin
from oneflow._C import atanh
from oneflow._C import atanh as arctanh
from oneflow._C import batch_matmul as bmm
from oneflow._C import baddbmm
from oneflow._C import broadcast_like
from oneflow._C import chunk
from oneflow._C import split
from oneflow._C import sign
from oneflow._C import sinh
from oneflow._C import tan
from oneflow._C import greater
from oneflow._C import greater as gt
from oneflow._C import greater_ as gt_
from oneflow._C import greater_equal
from oneflow._C import greater_equal as ge
from oneflow._C import log
from oneflow._C import log2
from oneflow._C import log10
from oneflow._C import logical_and
from oneflow._C import logical_or
from oneflow._C import logical_xor
from oneflow._C import logical_not
from oneflow._C import logaddexp
from oneflow._C import quantile
from oneflow._C import gelu_with_approximate as gelu
from oneflow._C import quick_gelu
from oneflow._C import mish
from oneflow._C import repeat
from oneflow._C import repeat_interleave
from oneflow._C import tile
from oneflow._C import sigmoid
from oneflow._C import tanh
from oneflow._C import as_strided
from oneflow._C import as_strided_
from oneflow._C import silu
from oneflow._C import selu
from oneflow._C import softshrink
from oneflow._C import softsign
from oneflow._C import cast
from oneflow._C import diag
from oneflow._C import log1p
from oneflow._C import add
from oneflow._C import addcdiv
from oneflow._C import div, div_
from oneflow._C import addcmul
from oneflow._C import floor, floor_
from oneflow._C import floor_divide
from oneflow._C import frac, frac_
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
from oneflow._C import ceil, ceil_
from oneflow._C import clamp, clamp_, clamp_min, clamp_min_, clamp_max, clamp_max_
from oneflow._C import clip, clip_
from oneflow._C import cos
from oneflow._C import cosh
from oneflow._C import diagonal
from oneflow._C import erf
from oneflow._C import erfc
from oneflow._C import expm1
from oneflow._C import fmod
from oneflow._C import flatten
from oneflow._C import topk
from oneflow._C import in_top_k
from oneflow._C import lgamma
from oneflow._C import minimum
from oneflow._C import maximum
from oneflow._C import max
from oneflow._C import min
from oneflow._C import median
from oneflow._C import mode
from oneflow._C import pow
from oneflow._C import reduce_prod as prod
from oneflow._C import reduce_sum as sum
from oneflow._C import reduce_mean as mean
from oneflow._C import reduce_all as all
from oneflow._C import reduce_any as any
from oneflow._C import reduce_nansum as nansum
from oneflow._C import logsumexp
from oneflow._C import rsqrt
from oneflow._C import sqrt
from oneflow._C import square
from oneflow._C import matmul
from oneflow._C import mm
from oneflow._C import matrix_vector_product as mv
from oneflow._C import bernoulli
from oneflow._C import round, round_
from oneflow._C import softplus
from oneflow._C import threshold
from oneflow._C import tril
from oneflow._C import triu
from oneflow._C import trunc
from oneflow._C import pad
from oneflow._C import transpose
from oneflow._C import relu
from oneflow._C import roc_auc_score
from oneflow._C import softmax
from oneflow._C import log_softmax
from oneflow._C import argmax
from oneflow._C import argmin
from oneflow._C import std
from oneflow._C import stft
from oneflow._C import var
from oneflow._C import stack, hstack, vstack, dstack, column_stack, row_stack
from oneflow._C import atleast_1d, atleast_2d, atleast_3d
from oneflow._C import squeeze
from oneflow._C import narrow
from oneflow._C import unsqueeze
from oneflow._C import permute
from oneflow._C import select
from oneflow._C import unbind
from oneflow._C import tensor_split
from oneflow._C import hann_window
from oneflow._C import hsplit
from oneflow._C import vsplit
from oneflow._C import concat
from oneflow._C import concat as cat
from oneflow._C import dim_gather as gather
from oneflow._C import deform_conv2d
from oneflow._C import gather_nd
from oneflow._C import roi_align
from oneflow._C import dot
from oneflow._C import eye
from oneflow._C import erfinv, erfinv_
from oneflow._C import cumsum
from oneflow._C import contiguous
from oneflow._C import cumprod
from oneflow._C import swapaxes
from oneflow._C import amax
from oneflow._C import swapdims
from oneflow._C import t
from oneflow._C import masked_fill
from oneflow._C import masked_fill_
from oneflow._C import equal
from oneflow._C import broadcast_equal as eq
from oneflow._C import not_equal
from oneflow._C import not_equal as ne
from oneflow._C import less as lt
from oneflow._C import less_equal as le
from oneflow._C import searchsorted
from oneflow._C import flip
from oneflow._C import index_select
from oneflow._C import isnan
from oneflow._C import isinf
from oneflow._C import isfinite
from oneflow._C import inv as inverse
from oneflow._C import det
from oneflow._C import iinfo, finfo
from oneflow._C import multinomial
from oneflow._C import linalg_cross as cross
from oneflow._C import bincount
from oneflow._C import isclose
from oneflow._C import allclose
from oneflow._C import lerp, lerp_
from oneflow._C import index_add, index_add_
from oneflow._C import sort
from oneflow._C import clone
from oneflow._C import bitwise_and, bitwise_or, bitwise_xor, bitwise_not

from oneflow._oneflow_internal import _set_num_threads as set_num_threads

from . import sbp

sbp.sbp.__call__ = lambda self: self

import atexit

import oneflow.framework.c_api_util
import oneflow.framework.register_class_method_util as register_class_method_util


register_class_method_util.RegisterMethod4Class()
import oneflow.framework.env_util as env_util
import oneflow.framework.scope_util as scope_util
import oneflow.framework.session_context as session_ctx
from oneflow.framework.tensor_str import set_printoptions

_oneflow_global_unique_env = env_util.GetEnv()
session_ctx.NewDefaultSession(_oneflow_global_unique_env)

oneflow._oneflow_internal.RegisterGILForeignLockHelper()
oneflow._oneflow_internal.autograd.graph.register_saved_tensors_hook_manager()
oneflow._oneflow_internal.RegisterStackGetter()


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
    _oneflow_global_unique_env.switch_to_shutting_down(hook.is_normal_exit())
    oneflow.framework.session_context.TryCloseDefaultSession()


atexit.register(atexit_hook, hook)
del atexit_hook
del hook
del ExitHook
del atexit
del oneflow

# default dtype
from oneflow.framework.dtype import (
    set_default_dtype,
    set_default_tensor_type,
    get_default_dtype,
    is_floating_point,
)

import oneflow._C
from oneflow._C import tensor, batch_gather
from oneflow._C import from_numpy, from_dlpack

from oneflow.autograd import (
    enable_grad,
    set_grad_enabled,
    no_grad,
    inference_mode,
    is_grad_enabled,
)
import oneflow.nn.image

from oneflow.framework.check_point_v2 import load
from oneflow.framework.check_point_v2 import save
from oneflow.framework.dtype import convert_oneflow_dtype_to_numpy_dtype, dtypes
from oneflow.framework.function_util import FunctionConfig
from oneflow.framework.function_util import FunctionConfig as function_config
from oneflow.framework.generator import create_generator as Generator
from oneflow.framework.generator import (
    default_generator,
    seed,
    manual_seed,
    initial_seed,
    get_rng_state,
    set_rng_state,
)

# NOTE(chengcheng) oneflow.Model is unavailable now.
# from oneflow.framework.model import Model
import oneflow.utils.tensor
import oneflow.utils.global_view
from oneflow.framework.tensor import Tensor
from oneflow.framework.tensor import is_nonzero
from oneflow._oneflow_internal import to_dlpack
from oneflow.framework.type_tensor import *

from oneflow.framework.tensor import zero_

from oneflow.nn.modules.pooling import (
    adaptive_avg_pool1d,
    adaptive_avg_pool2d,
    adaptive_avg_pool3d,
)
from oneflow.nn.modules.einsum import einsum_op as einsum
from oneflow.nn.modules.is_tensor import is_tensor_op as is_tensor
from oneflow.nn.modules.arange import arange_op as arange
from oneflow.nn.modules.linspace import linspace_op as linspace
from oneflow.nn.modules.logspace import logspace_op as logspace
from oneflow.nn.modules.argsort import argsort_op as argsort
from oneflow.nn.modules.argwhere import argwhere_op as argwhere
from oneflow.nn.modules.constant import ones_op as ones
from oneflow.nn.modules.constant import zeros_op as zeros
from oneflow.nn.modules.constant import zeros_like_op as zeros_like
from oneflow.nn.modules.constant import ones_like_op as ones_like
from oneflow.nn.modules.constant import full_op as full
from oneflow.nn.modules.constant import full_like_op as full_like
from oneflow.nn.modules.constant import new_ones_op as new_ones
from oneflow.nn.modules.constant import new_zeros_op as new_zeros
from oneflow.nn.modules.constant import new_full_op as new_full
from oneflow.nn.modules.empty import empty_op as empty
from oneflow.nn.modules.empty import new_empty_op as new_empty
from oneflow.nn.modules.empty import empty_like_op as empty_like
from oneflow._C import empty_strided
from oneflow.nn.modules.dataset import tensor_buffer_to_list_of_tensors
from oneflow._C import movedim
from oneflow.nn.modules.expand import expand_op as expand
from oneflow.nn.modules.distributed_partial_fc_sample import (
    distributed_partial_fc_sample_op as distributed_partial_fc_sample,
)
from oneflow.nn.modules.roll import roll_op as roll
from oneflow.nn.modules.masked_select import masked_select_op as masked_select
from oneflow.nn.modules.math_ops import addmm_op as addmm
from oneflow.nn.modules.nonzero import nonzero_op as nonzero
from oneflow.nn.modules.nms import nms_op as nms
from oneflow.nn.modules.numel import numel_op as numel
from oneflow.nn.modules.meshgrid import meshgrid_op as meshgrid
from oneflow.nn.modules.unique import unique_op as unique
from oneflow._C import normal
from oneflow._C import rand
from oneflow._C import randn
from oneflow._C import randn_like
from oneflow._C import randint
from oneflow._C import randint_like
from oneflow._C import randperm
from oneflow.nn.modules.reshape import reshape_op as reshape
from oneflow.nn.modules.reshape import view_op as view
from oneflow.nn.modules.slice import slice_op as slice
from oneflow.nn.modules.slice import slice_update_op as slice_update
from oneflow.nn.modules.tensor_buffer import gen_tensor_buffer
from oneflow.nn.modules.tensor_buffer import (
    tensor_buffer_to_tensor_op as tensor_buffer_to_tensor,
)
from oneflow.nn.modules.tensordot import tensordot
from oneflow.nn.modules.norm import norm
from oneflow.nn.modules.as_tensor import as_tensor
from oneflow.nn.modules.tensor_buffer import tensor_to_tensor_buffer
from oneflow.nn.modules.global_cast import local_to_global_op as local_to_global
from oneflow.nn.modules.global_cast import global_to_global_op as global_to_global
from oneflow.nn.modules.global_cast import to_global_op as to_global
from oneflow.nn.modules.global_cast import to_local_op as to_local
from oneflow.nn.modules.where import where_op as where
from oneflow.nn.modules.scatter import *
from oneflow.nn.modules.broadcast_ops import (
    broadcast_tensors,
    broadcast_shapes,
    broadcast_to,
)
from oneflow.ops.stateful_ops import StatefulOp as stateful_op

# autocast
from oneflow._oneflow_internal import (
    is_autocast_enabled,
    set_autocast_enabled,
    get_autocast_gpu_dtype,
    get_autocast_cpu_dtype,
    set_autocast_gpu_dtype,
    set_autocast_cpu_dtype,
    is_autocast_cache_enabled,
    set_autocast_cache_enabled,
    clear_autocast_cache,
)
from oneflow.amp.autocast_mode import *
from oneflow.jit import *

from . import (
    autograd,
    distributed,
    distributions,
    linalg,
    optim,
    comm,
    boxing,
    backends,
    amp,
    hub,
    fx,
    special,
)
import oneflow.utils.data
import oneflow.framework.docstr as docstr
import oneflow.cuda
import oneflow.multiprocessing
import oneflow.asyncs
import oneflow.one_embedding
import oneflow.profiler
import oneflow.mock_torch

if oneflow._oneflow_internal.flags.with_mlir():
    oneflow_internal_path = oneflow._oneflow_internal.__file__
    if os.getenv("ONEFLOW_MLIR_ENABLE_CODEGEN_FUSERS") or os.getenv(
        "ONEFLOW_MLIR_FUSE_KERNEL_LAUNCH"
    ):
        print("MLIR JIT engine will load:", oneflow_internal_path, file=sys.stderr)
        oneflow._oneflow_internal.ir.load_jit_shared_lib(oneflow_internal_path)
