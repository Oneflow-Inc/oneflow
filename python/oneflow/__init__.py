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
from oneflow.version import __version__
from oneflow.core.job.job_set_pb2 import ConfigProto
from oneflow.core.job.job_conf_pb2 import JobConfigProto

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


import oneflow.framework.register_python_callback
import atexit
import oneflow.framework.c_api_util
import oneflow.framework.register_class_method_util as register_class_method_util

INVALID_SPLIT_AXIS = oneflow._oneflow_internal.INVALID_SPLIT_AXIS
register_class_method_util.RegisterMethod4Class()
oneflow._oneflow_internal.RegisterGILForeignLockHelper()
import oneflow.framework.env_util as env_util
import oneflow.framework.session_context as session_ctx
import oneflow.framework.scope_util as scope_util
from oneflow.framework.session_util import Session
from oneflow.framework.multi_client_session import MultiClientSession

if not env_util.HasAllMultiClientEnvVars():
    env_util.SetDefaultMultiClientEnvVars()
oneflow._oneflow_internal.SetIsMultiClient(True)
env_util.api_env_init()
session_ctx.OpenDefaultSession(
    MultiClientSession(oneflow._oneflow_internal.NewSessionId())
)
scope_util.InitScopeStack()
oneflow._oneflow_internal.EnableEagerEnvironment(True)
del env_util


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
from oneflow.ops.constant_op import constant
from oneflow.ops.constant_op import constant_scalar
from oneflow.ops.constant_op import constant_like
from oneflow.ops.categorical_ordinal_encode_op import categorical_ordinal_encode
from oneflow.ops.array_ops import reshape_like
from oneflow.ops.array_ops import dynamic_reshape
from oneflow.ops.array_ops import slice_v2
from oneflow.ops.array_ops import reverse
from oneflow.ops.array_ops import concat
from oneflow.ops.array_ops import tensor_scatter_nd_update
from oneflow.ops.array_ops import tensor_scatter_nd_add
from oneflow.ops.array_ops import elem_cnt
from oneflow.ops.array_ops import sync_dynamic_resize
from oneflow.ops.array_ops import identity
from oneflow.ops.array_ops import identity_n
from oneflow.ops.array_ops import cast_to_static_shape
from oneflow.ops.array_ops import expand_dims
from oneflow.ops.array_ops import dim_gather
from oneflow.ops.array_ops import amp_white_identity
from oneflow.ops.one_hot import one_hot
from oneflow.ops.combined_margin_loss import combined_margin_loss
from oneflow.ops.assign_op import assign
from oneflow.ops.pad import pad
from oneflow.ops.pad import pad_grad
from oneflow.ops.pad import same_padding
from oneflow.ops.pad import replication_pad2d
from oneflow.ops.pad import constant_pad2d
from oneflow.ops.pad import zero_pad2d
from oneflow.ops.builtin_ops import BuiltinOp
from oneflow.ops.initializer_util import empty_initializer
from oneflow.ops.initializer_util import constant_initializer
from oneflow.ops.initializer_util import zeros_initializer
from oneflow.ops.initializer_util import ones_initializer
from oneflow.ops.initializer_util import random_uniform_initializer
from oneflow.ops.initializer_util import random_normal_initializer
from oneflow.ops.initializer_util import truncated_normal_initializer
from oneflow.ops.initializer_util import (
    glorot_uniform_initializer as xavier_uniform_initializer,
)
from oneflow.ops.initializer_util import glorot_uniform_initializer
from oneflow.ops.initializer_util import (
    glorot_normal_initializer as xavier_normal_initializer,
)
from oneflow.ops.initializer_util import glorot_normal_initializer
from oneflow.ops.initializer_util import variance_scaling_initializer
from oneflow.ops.initializer_util import kaiming_initializer
from oneflow.ops.math_ops import unsorted_segment_sum as unsorted_segment_sum
from oneflow.ops.math_ops import unsorted_segment_sum_like as unsorted_segment_sum_like
from oneflow.ops.math_ops import (
    unsorted_batch_segment_sum as unsorted_batch_segment_sum,
)
from oneflow.ops.math_ops import (
    broadcast_to_compatible_with as broadcast_to_compatible_with,
)
from oneflow.ops.user_data_ops import api_image_resize as image_resize
from oneflow.ops.user_data_ops import api_image_target_resize as image_target_resize
from oneflow.ops.user_data_ops import api_image_random_crop as image_random_crop
from oneflow.ops.user_data_ops import image_decode as image_decode
from oneflow.ops.user_data_ops import image_batch_align as image_batch_align
from oneflow.ops.user_data_ops import image_normalize as image_normalize
from oneflow.ops.user_data_ops import image_flip as image_flip
from oneflow.ops.user_data_ops import object_bbox_flip as object_bbox_flip
from oneflow.ops.user_data_ops import object_bbox_scale as object_bbox_scale
from oneflow.ops.user_data_ops import (
    object_segm_poly_flip as object_segmentation_polygon_flip,
)
from oneflow.ops.user_data_ops import (
    object_segm_poly_scale as object_segmentation_polygon_scale,
)
from oneflow.ops.user_data_ops import (
    object_segm_poly_to_mask as object_segmentation_polygon_to_mask,
)
from oneflow.ops.watch import Watch
from oneflow.ops.watch import WatchDiff
from oneflow.ops.user_op_builder import api_user_op_builder
from oneflow.ops.user_op_builder import api_consistent_user_op_builder
from oneflow.ops.user_op_builder import api_user_op_module_builder
from oneflow.ops.user_op_builder import api_consistent_user_op_module_builder
from oneflow.ops.eager_nccl_ops import eager_nccl_all_reduce
from oneflow.ops.count_not_finite import count_not_finite
from oneflow.ops.count_not_finite import multi_count_not_finite
from oneflow.ops.partial_fc_sample import distributed_partial_fc_sample
from oneflow.ops.loss_ops import smooth_l1_loss
from oneflow.ops.loss_ops import ctc_loss
from oneflow.ops.get_variable import api_get_variable
from oneflow.nn.modules.sign import sign_op
from oneflow.nn.modules.meshgrid import meshgrid_op
from oneflow.nn.modules.reshape import reshape_op
from oneflow.nn.modules.reshape import view_op
from oneflow.nn.modules.greater import greater_op
from oneflow.nn.modules.floor import floor_op
from oneflow.nn.modules.acosh import acosh_op
from oneflow.nn.modules.acosh import arccosh_op
from oneflow.nn.modules.stack import stack
from oneflow.nn.modules.sinh import sinh_op
from oneflow.nn.modules.to import to_op
from oneflow.nn.modules.expand import expand_op
from oneflow.nn.modules.diag import diag_op
from oneflow.nn.modules.exp import exp_op
from oneflow.nn.modules.argwhere import argwhere_op
from oneflow.nn.modules.concat import concat_op
from oneflow.nn.modules.argmax import argmax_op
from oneflow.nn.modules.greater_equal import greater_equal_op
from oneflow.nn.modules.math_ops import _mul
from oneflow.nn.modules.math_ops import variance_op
from oneflow.nn.modules.math_ops import _sub
from oneflow.nn.modules.math_ops import _div
from oneflow.nn.modules.math_ops import _reciprocal
from oneflow.nn.modules.math_ops import _add
from oneflow.nn.modules.math_ops import asin_op
from oneflow.nn.modules.math_ops import arcsin_op
from oneflow.nn.modules.math_ops import asinh_op
from oneflow.nn.modules.math_ops import arcsinh_op
from oneflow.nn.modules.math_ops import sin_op
from oneflow.nn.modules.math_ops import cos_op
from oneflow.nn.modules.math_ops import atan_op
from oneflow.nn.modules.math_ops import arctan_op
from oneflow.nn.modules.math_ops import fmod_op
from oneflow.nn.modules.math_ops import log_op
from oneflow.nn.modules.math_ops import rsqrt_op
from oneflow.nn.modules.math_ops import sqrt_op
from oneflow.nn.modules.math_ops import square_op
from oneflow.nn.modules.math_ops import std_op
from oneflow.nn.modules.math_ops import pow_op
from oneflow.nn.modules.math_ops import addmm_op
from oneflow.nn.modules.math_ops import clamp_op
from oneflow.nn.modules.math_ops import clip_op
from oneflow.nn.modules.math_ops import cosh_op
from oneflow.nn.modules.math_ops import erf_op
from oneflow.nn.modules.math_ops import erfc_op
from oneflow.nn.modules.math_ops import ceil_op
from oneflow.nn.modules.math_ops import expm1_op
from oneflow.nn.modules.math_ops import topk_op
from oneflow.nn.modules.transpose import transpose_op
from oneflow.nn.modules.in_top_k import in_top_k_op
from oneflow.nn.modules.adaptive_pool import adaptive_avg_pool1d
from oneflow.nn.modules.adaptive_pool import adaptive_avg_pool2d
from oneflow.nn.modules.adaptive_pool import adaptive_avg_pool3d
from oneflow.nn.modules.random_ops import bernoulli
from oneflow.nn.modules.abs import abs_op
from oneflow.nn.modules.less import less_op
from oneflow.nn.modules.atan2 import atan2_op
from oneflow.nn.modules.gather import gather_op
from oneflow.nn.modules.less_equal import less_equal_op
from oneflow.nn.modules.eq import eq_op as equal
from oneflow.nn.modules.eq import eq_op
from oneflow.nn.modules.masked_fill import masked_fill_op
from oneflow.nn.modules.dataset import tensor_buffer_to_list_of_tensors
from oneflow.nn.modules.flatten import _flow_flatten
from oneflow.nn.modules.tile import tile_op
from oneflow.nn.modules.triu import triu_op
from oneflow.nn.modules.tensor_buffer import tensor_buffer_to_tensor_op
from oneflow.nn.modules.tensor_buffer import tensor_to_tensor_buffer
from oneflow.nn.modules.tensor_buffer import gen_tensor_buffer
from oneflow.nn.modules.argsort import argsort_op
from oneflow.nn.modules.arange import arange_op
from oneflow.nn.modules.ne import ne_op as not_equal
from oneflow.nn.modules.ne import ne_op
from oneflow.nn.modules.slice import slice_op
from oneflow.nn.modules.slice import slice_update_op
from oneflow.nn.modules.acos import acos_op
from oneflow.nn.modules.squeeze import squeeze_op
from oneflow.nn.modules.log1p import log1p_op
from oneflow.nn.modules.scatter_nd import Scatter_nd
from oneflow.nn.modules.broadcast_like import broadcast_like_op
from oneflow.nn.modules.tan import tan_op
from oneflow.nn.modules.cast import cast_op
from oneflow.nn.modules.sort import sort_op
from oneflow.nn.modules.flip import flip_op
from oneflow.nn.modules.reduce_ops import _sum
from oneflow.nn.modules.reduce_ops import _mean
from oneflow.nn.modules.reduce_ops import _min
from oneflow.nn.modules.reduce_ops import _max
from oneflow.nn.modules.chunk import chunk_op
from oneflow.nn.modules.unsqueeze import unsqueeze_op
from oneflow.nn.modules.activation import tanh_op
from oneflow.nn.modules.activation import gelu_op
from oneflow.nn.modules.activation import sigmoid_op
from oneflow.nn.modules.activation import softmax_op
from oneflow.nn.modules.activation import mish_op
from oneflow.nn.modules.repeat import repeat_op
from oneflow.nn.modules.round import round_op
from oneflow.nn.modules.constant import ones_op
from oneflow.nn.modules.constant import zeros_op
from oneflow.nn.modules.constant import zeros_like_op
from oneflow.nn.modules.constant import ones_like_op
from oneflow.nn.modules.bmm import bmm_op
from oneflow.nn.modules.gather_nd import gather_nd_op
from oneflow.nn.modules.softplus import softplus_op
from oneflow.nn.modules.negative import negative_op as neg
from oneflow.nn.modules.negative import negative_op
from oneflow.nn.modules.matmul import matmul_op
from oneflow.nn.modules.where import where_op
from oneflow.nn.modules.atanh import atanh_op
from oneflow.nn.modules.atanh import arctanh_op
from oneflow.nn.modules.masked_select import masked_select_op
from oneflow.deprecated.initializer_util import truncated_normal_initializer
from oneflow.advanced.distribute_ops import cast_to_current_logical_view
from oneflow.experimental.name_scope import deprecated_name_scope
from oneflow.framework.session_util import api_find_or_create_module
from oneflow.framework.session_util import api_eager_execution_enabled
from oneflow.framework.session_util import api_clear_default_session
from oneflow.framework.session_util import api_sync_default_session
from oneflow.framework.session_util import TmpInitEagerGlobalSession
from oneflow.framework.tensor import construct_tensor
from oneflow.framework.tensor import Tensor
from oneflow.framework.check_point_v2 import GetAllVariables
from oneflow.framework.check_point_v2 import Load
from oneflow.framework.check_point_v2 import save
from oneflow.framework.check_point_v2 import LoadVariables
from oneflow.framework.scope_util import api_current_scope
from oneflow.framework.generator import create_generator
from oneflow.framework.generator import default_generator
from oneflow.framework.generator import manual_seed
from oneflow.framework.model import Model
from oneflow.framework.function_util import FunctionConfig as function_config
from oneflow.framework.function_util import FunctionConfig as ExecutionConfig
from oneflow.framework.function_util import FunctionConfig
from oneflow.framework.function_util import api_oneflow_function
from oneflow.framework.job_set_util import inter_job_reuse_mem_strategy
from oneflow.framework.function_desc import api_current_global_function_desc
from oneflow.framework.ops import api_acc
from oneflow.framework.ops import api_unpack
from oneflow.framework.ops import api_pack
from oneflow.framework.ops import api_parallel_cast
from oneflow.framework.ops import api_hierarchical_parallel_cast
from oneflow.framework.env_util import api_enable_eager_execution
from oneflow.framework.env_util import api_get_current_resource as current_resource
from oneflow.framework.env_util import api_get_current_machine_id
from oneflow.framework.placement_util import deprecated_placement as fixed_placement
from oneflow.framework.placement_util import deprecated_placement
from oneflow.framework.dtype import dtypes
from oneflow.framework.dtype import convert_oneflow_dtype_to_numpy_dtype
from oneflow.framework.input_blob_def import DeprecatedFixedTensorDef
from oneflow.framework.input_blob_def import DeprecatedMirroredTensorDef
