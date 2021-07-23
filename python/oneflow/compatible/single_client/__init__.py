import oneflow._oneflow_internal

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
from oneflow.core.job.job_set_pb2 import ConfigProto
from oneflow.core.job.job_conf_pb2 import JobConfigProto
from oneflow.compatible.single_client.python.framework import session_util
from oneflow.compatible.single_client.python.framework import session_context
from oneflow.compatible.single_client.python.framework import env_util

oneflow._oneflow_internal.DestroyEnv()
import time

time.sleep(1)
del time
oneflow._oneflow_internal.SetIsMultiClient(False)
session_context.OpenDefaultSession(
    session_util.Session(oneflow._oneflow_internal.NewSessionId())
)
oneflow._oneflow_internal.EnableEagerEnvironment(False)
del env_util
del session_util
del session_context
import oneflow.compatible.single_client.python.framework.c_api_util
from oneflow.compatible.single_client.python.framework import register_python_callback
from oneflow.compatible.single_client.python.framework import python_callback

oneflow._oneflow_internal.RegisterForeignCallbackOnlyOnce(
    python_callback.global_python_callback
)
del python_callback
del register_python_callback
from oneflow.compatible.single_client.python.framework import watcher

oneflow._oneflow_internal.RegisterWatcherOnlyOnce(watcher._global_watcher)
del watcher
from oneflow.compatible.single_client.python.eager import boxing_util

oneflow._oneflow_internal.deprecated.RegisterBoxingUtilOnlyOnce(
    boxing_util._global_boxing_util
)
del boxing_util
from oneflow.compatible.single_client.python.ops.util import custom_op_module

oneflow._oneflow_internal.RegisterPyKernels(
    custom_op_module._python_kernel_reg.kernels_
)
del custom_op_module
from oneflow.compatible.single_client.python.framework import register_class_method_util

register_class_method_util.RegisterMethod4Class()
del register_class_method_util
INVALID_SPLIT_AXIS = oneflow._oneflow_internal.INVALID_SPLIT_AXIS
import atexit
from oneflow.compatible.single_client.python.framework.session_context import (
    TryCloseAllSession,
)

atexit.register(TryCloseAllSession)
del TryCloseAllSession
del atexit
import sys

__original_exit__ = sys.exit


def custom_exit(returncode):
    if returncode != 0:
        import oneflow

        oneflow._oneflow_internal.MasterSendAbort()
    __original_exit__(returncode)


sys.exit = custom_exit
del custom_exit
del sys
from oneflow.compatible.single_client.ops.constant_op import constant
from oneflow.compatible.single_client.ops.constant_op import constant_scalar
from oneflow.compatible.single_client.ops.constant_op import constant_like
from oneflow.compatible.single_client.ops.constant_op import ones_like
from oneflow.compatible.single_client.ops.constant_op import zeros_like
from oneflow.compatible.single_client.ops.categorical_ordinal_encode_op import (
    categorical_ordinal_encode,
)
from oneflow.compatible.single_client.ops.array_ops import gather
from oneflow.compatible.single_client.ops.array_ops import flatten
from oneflow.compatible.single_client.ops.array_ops import reshape
from oneflow.compatible.single_client.ops.array_ops import reshape_like
from oneflow.compatible.single_client.ops.array_ops import dynamic_reshape
from oneflow.compatible.single_client.ops.array_ops import transpose
from oneflow.compatible.single_client.ops.array_ops import slice
from oneflow.compatible.single_client.ops.array_ops import slice_v2
from oneflow.compatible.single_client.ops.array_ops import api_slice_update
from oneflow.compatible.single_client.ops.array_ops import reverse
from oneflow.compatible.single_client.ops.array_ops import concat
from oneflow.compatible.single_client.ops.array_ops import gather_nd
from oneflow.compatible.single_client.ops.array_ops import scatter_nd
from oneflow.compatible.single_client.ops.array_ops import tensor_scatter_nd_update
from oneflow.compatible.single_client.ops.array_ops import tensor_scatter_nd_add
from oneflow.compatible.single_client.ops.array_ops import argwhere
from oneflow.compatible.single_client.ops.array_ops import nonzero
from oneflow.compatible.single_client.ops.array_ops import where
from oneflow.compatible.single_client.ops.array_ops import elem_cnt
from oneflow.compatible.single_client.ops.array_ops import sync_dynamic_resize
from oneflow.compatible.single_client.ops.array_ops import stack
from oneflow.compatible.single_client.ops.array_ops import identity
from oneflow.compatible.single_client.ops.array_ops import identity_n
from oneflow.compatible.single_client.ops.array_ops import cast_to_static_shape
from oneflow.compatible.single_client.ops.array_ops import squeeze
from oneflow.compatible.single_client.ops.array_ops import expand
from oneflow.compatible.single_client.ops.array_ops import expand_dims
from oneflow.compatible.single_client.ops.array_ops import broadcast_like
from oneflow.compatible.single_client.ops.array_ops import masked_fill
from oneflow.compatible.single_client.ops.array_ops import dim_gather
from oneflow.compatible.single_client.ops.array_ops import amp_white_identity
from oneflow.compatible.single_client.ops.array_ops import zeros
from oneflow.compatible.single_client.ops.array_ops import ones
from oneflow.compatible.single_client.ops.one_hot import one_hot
from oneflow.compatible.single_client.ops.linalg import matmul
from oneflow.compatible.single_client.ops.combined_margin_loss import (
    combined_margin_loss,
)
from oneflow.compatible.single_client.ops.assign_op import assign
from oneflow.compatible.single_client.ops.pad import pad
from oneflow.compatible.single_client.ops.pad import pad_grad
from oneflow.compatible.single_client.ops.pad import same_padding
from oneflow.compatible.single_client.ops.pad import reflection_pad2d
from oneflow.compatible.single_client.ops.pad import replication_pad2d
from oneflow.compatible.single_client.ops.pad import constant_pad2d
from oneflow.compatible.single_client.ops.pad import zero_pad2d
from oneflow.compatible.single_client.ops.builtin_ops import BuiltinOp
from oneflow.compatible.single_client.ops.initializer_util import empty_initializer
from oneflow.compatible.single_client.ops.initializer_util import constant_initializer
from oneflow.compatible.single_client.ops.initializer_util import zeros_initializer
from oneflow.compatible.single_client.ops.initializer_util import ones_initializer
from oneflow.compatible.single_client.ops.initializer_util import (
    random_uniform_initializer,
)
from oneflow.compatible.single_client.ops.initializer_util import (
    random_normal_initializer,
)
from oneflow.compatible.single_client.ops.initializer_util import (
    truncated_normal_initializer,
)
from oneflow.compatible.single_client.ops.initializer_util import (
    glorot_uniform_initializer as xavier_uniform_initializer,
)
from oneflow.compatible.single_client.ops.initializer_util import (
    glorot_uniform_initializer,
)
from oneflow.compatible.single_client.ops.initializer_util import (
    glorot_normal_initializer as xavier_normal_initializer,
)
from oneflow.compatible.single_client.ops.initializer_util import (
    glorot_normal_initializer,
)
from oneflow.compatible.single_client.ops.initializer_util import (
    variance_scaling_initializer,
)
from oneflow.compatible.single_client.ops.initializer_util import kaiming_initializer
from oneflow.compatible.single_client.ops.math_ops import (
    unsorted_segment_sum as unsorted_segment_sum,
)
from oneflow.compatible.single_client.ops.math_ops import (
    unsorted_segment_sum_like as unsorted_segment_sum_like,
)
from oneflow.compatible.single_client.ops.math_ops import (
    unsorted_batch_segment_sum as unsorted_batch_segment_sum,
)
from oneflow.compatible.single_client.ops.math_ops import cast
from oneflow.compatible.single_client.ops.math_ops import (
    broadcast_to_compatible_with as broadcast_to_compatible_with,
)
from oneflow.compatible.single_client.ops.math_ops import clip_by_value as clip_by_value
from oneflow.compatible.single_client.ops.math_ops import (
    clip_by_value as clip_by_scalar,
)
from oneflow.compatible.single_client.ops.math_ops import clip_by_value as clip
from oneflow.compatible.single_client.ops.math_ops import clip_by_value as clamp
from oneflow.compatible.single_client.ops.math_ops import in_top_k as in_top_k
from oneflow.compatible.single_client.ops.math_ops import range
from oneflow.compatible.single_client.ops.user_data_ops import (
    api_image_resize as image_resize,
)
from oneflow.compatible.single_client.ops.user_data_ops import (
    api_image_target_resize as image_target_resize,
)
from oneflow.compatible.single_client.ops.user_data_ops import (
    api_image_random_crop as image_random_crop,
)
from oneflow.compatible.single_client.ops.user_data_ops import (
    image_decode as image_decode,
)
from oneflow.compatible.single_client.ops.user_data_ops import (
    image_batch_align as image_batch_align,
)
from oneflow.compatible.single_client.ops.user_data_ops import (
    image_normalize as image_normalize,
)
from oneflow.compatible.single_client.ops.user_data_ops import image_flip as image_flip
from oneflow.compatible.single_client.ops.user_data_ops import (
    object_bbox_flip as object_bbox_flip,
)
from oneflow.compatible.single_client.ops.user_data_ops import (
    object_bbox_scale as object_bbox_scale,
)
from oneflow.compatible.single_client.ops.user_data_ops import (
    object_segm_poly_flip as object_segmentation_polygon_flip,
)
from oneflow.compatible.single_client.ops.user_data_ops import (
    object_segm_poly_scale as object_segmentation_polygon_scale,
)
from oneflow.compatible.single_client.ops.user_data_ops import (
    object_segm_poly_to_mask as object_segmentation_polygon_to_mask,
)
from oneflow.compatible.single_client.ops.watch import Watch
from oneflow.compatible.single_client.ops.watch import WatchDiff
from oneflow.compatible.single_client.ops.user_op_builder import api_user_op_builder
from oneflow.compatible.single_client.ops.user_op_builder import (
    api_consistent_user_op_builder,
)
from oneflow.compatible.single_client.ops.user_op_builder import (
    api_user_op_module_builder,
)
from oneflow.compatible.single_client.ops.user_op_builder import (
    api_consistent_user_op_module_builder,
)
from oneflow.compatible.single_client.ops.diag_ops import diag
from oneflow.compatible.single_client.ops.eager_nccl_ops import eager_nccl_all_reduce
from oneflow.compatible.single_client.ops.count_not_finite import count_not_finite
from oneflow.compatible.single_client.ops.count_not_finite import multi_count_not_finite
from oneflow.compatible.single_client.ops.sort_ops import sort
from oneflow.compatible.single_client.ops.sort_ops import argsort
from oneflow.compatible.single_client.ops.partial_fc_sample import (
    distributed_partial_fc_sample,
)
from oneflow.compatible.single_client.ops.tensor_buffer_ops import (
    tensor_buffer_to_tensor,
)
from oneflow.compatible.single_client.ops.tensor_buffer_ops import (
    tensor_to_tensor_buffer,
)
from oneflow.compatible.single_client.ops.tensor_buffer_ops import gen_tensor_buffer
from oneflow.compatible.single_client.ops.tensor_buffer_ops import (
    tensor_buffer_to_list_of_tensors,
)
from oneflow.compatible.single_client.ops.loss_ops import smooth_l1_loss
from oneflow.compatible.single_client.ops.loss_ops import ctc_loss
from oneflow.compatible.single_client.ops.get_variable import api_get_variable
from oneflow.compatible.single_client.nn.modules.to import to_op
from oneflow.compatible.single_client.deprecated.initializer_util import (
    truncated_normal_initializer,
)
from oneflow.compatible.single_client.advanced.distribute_ops import (
    cast_to_current_logical_view,
)
from oneflow.compatible.single_client.experimental.name_scope import (
    deprecated_name_scope,
)
from oneflow.compatible.single_client.framework.session_util import (
    api_find_or_create_module,
)
from oneflow.compatible.single_client.framework.session_util import (
    api_eager_execution_enabled,
)
from oneflow.compatible.single_client.framework.session_util import (
    api_clear_default_session,
)
from oneflow.compatible.single_client.framework.session_util import (
    api_sync_default_session,
)
from oneflow.compatible.single_client.framework.session_util import (
    TmpInitEagerGlobalSession,
)
from oneflow.compatible.single_client.framework.tensor import construct_tensor
from oneflow.compatible.single_client.framework.tensor import Tensor
from oneflow.compatible.single_client.framework.check_point_v2 import GetAllVariables
from oneflow.compatible.single_client.framework.check_point_v2 import Load
from oneflow.compatible.single_client.framework.check_point_v2 import save
from oneflow.compatible.single_client.framework.check_point_v2 import LoadVariables
from oneflow.compatible.single_client.framework.scope_util import api_current_scope
from oneflow.compatible.single_client.framework.generator import create_generator
from oneflow.compatible.single_client.framework.generator import default_generator
from oneflow.compatible.single_client.framework.generator import manual_seed
from oneflow.compatible.single_client.framework.model import Model
from oneflow.compatible.single_client.framework.function_util import (
    FunctionConfig as function_config,
)
from oneflow.compatible.single_client.framework.function_util import (
    FunctionConfig as ExecutionConfig,
)
from oneflow.compatible.single_client.framework.function_util import FunctionConfig
from oneflow.compatible.single_client.framework.function_util import (
    api_oneflow_function,
)
from oneflow.compatible.single_client.framework.job_set_util import (
    inter_job_reuse_mem_strategy,
)
from oneflow.compatible.single_client.framework.function_desc import (
    api_current_global_function_desc,
)
from oneflow.compatible.single_client.framework.ops import api_repeat
from oneflow.compatible.single_client.framework.ops import api_acc
from oneflow.compatible.single_client.framework.ops import api_unpack
from oneflow.compatible.single_client.framework.ops import api_pack
from oneflow.compatible.single_client.framework.ops import api_parallel_cast
from oneflow.compatible.single_client.framework.ops import (
    api_hierarchical_parallel_cast,
)
from oneflow.compatible.single_client.framework.env_util import (
    api_enable_eager_execution,
)
from oneflow.compatible.single_client.framework.env_util import (
    api_get_current_resource as current_resource,
)
from oneflow.compatible.single_client.framework.env_util import (
    api_get_current_machine_id,
)
from oneflow.compatible.single_client.framework.placement_util import (
    deprecated_placement as fixed_placement,
)
from oneflow.compatible.single_client.framework.placement_util import (
    deprecated_placement,
)
from oneflow.compatible.single_client.framework.dtype import dtypes
from oneflow.compatible.single_client.framework.dtype import (
    convert_oneflow_dtype_to_numpy_dtype,
)
from oneflow.compatible.single_client.framework.input_blob_def import (
    DeprecatedFixedTensorDef,
)
from oneflow.compatible.single_client.framework.input_blob_def import (
    DeprecatedMirroredTensorDef,
)
