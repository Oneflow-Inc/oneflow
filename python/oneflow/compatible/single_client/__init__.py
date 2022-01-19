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

import oneflow._oneflow_internal

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
locals()["uint16"] = oneflow._oneflow_internal.uint16
locals()["uint32"] = oneflow._oneflow_internal.uint32
locals()["uint64"] = oneflow._oneflow_internal.uint64
locals()["uint128"] = oneflow._oneflow_internal.uint128
locals()["int16"] = oneflow._oneflow_internal.int16
locals()["int128"] = oneflow._oneflow_internal.int128
locals()["complex32"] = oneflow._oneflow_internal.complex32
locals()["complex64"] = oneflow._oneflow_internal.complex64
locals()["complex128"] = oneflow._oneflow_internal.complex128
from oneflow.compatible.single_client.framework import (
    env_util,
    session_context,
    session_util,
)
from oneflow.core.job.job_conf_pb2 import JobConfigProto
from oneflow.core.job.job_set_pb2 import ConfigProto

oneflow._oneflow_internal.DestroyGlobalForeignCallback()
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
import oneflow.compatible.single_client.framework.c_api_util
from oneflow.compatible.single_client.framework import (
    python_callback,
    register_python_callback,
)

oneflow._oneflow_internal.RegisterGlobalForeignCallback(
    python_callback.global_python_callback
)
del python_callback
del register_python_callback
from oneflow.compatible.single_client.framework import watcher

oneflow._oneflow_internal.RegisterGlobalWatcher(watcher._global_watcher)
del watcher
from oneflow.compatible.single_client.eager import boxing_util

oneflow._oneflow_internal.deprecated.RegisterBoxingUtilOnlyOnce(
    boxing_util._global_boxing_util
)
del boxing_util
from oneflow.compatible.single_client.ops.util import custom_op_module

oneflow._oneflow_internal.RegisterPyKernels(
    custom_op_module._python_kernel_reg.kernels_
)
del custom_op_module
from oneflow.compatible.single_client.framework import register_class_method_util

register_class_method_util.RegisterMethod4Class()
del register_class_method_util
INVALID_SPLIT_AXIS = oneflow._oneflow_internal.INVALID_SPLIT_AXIS
import atexit

from oneflow.compatible.single_client.framework.session_context import (
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
from oneflow.compatible.single_client.autograd import no_grad
from oneflow.compatible.single_client.advanced.distribute_ops import (
    cast_to_current_logical_view,
)
from oneflow.compatible.single_client.deprecated.initializer_util import (
    truncated_normal_initializer as truncated_normal,
)
from oneflow.compatible.single_client.experimental.namescope import (
    deprecated_name_scope as name_scope,
)
from oneflow.compatible.single_client.framework.check_point_v2 import (
    GetAllVariables as get_all_variables,
)
from oneflow.compatible.single_client.framework.check_point_v2 import Load as load
from oneflow.compatible.single_client.framework.check_point_v2 import (
    LoadVariables as load_variables,
)
from oneflow.compatible.single_client.framework.check_point_v2 import save
from oneflow.compatible.single_client.framework.dtype import (
    convert_oneflow_dtype_to_numpy_dtype,
    dtypes,
)
from oneflow.compatible.single_client.framework.env_util import (
    api_enable_eager_execution as enable_eager_execution,
)
from oneflow.compatible.single_client.framework.env_util import (
    api_get_current_machine_id as current_machine_id,
)
from oneflow.compatible.single_client.framework.env_util import (
    api_get_current_resource as current_resource,
)
from oneflow.compatible.single_client.framework.function_desc import (
    api_current_global_function_desc as current_global_function_desc,
)
from oneflow.compatible.single_client.framework.function_util import FunctionConfig
from oneflow.compatible.single_client.framework.function_util import (
    FunctionConfig as ExecutionConfig,
)
from oneflow.compatible.single_client.framework.function_util import (
    FunctionConfig as function_config,
)
from oneflow.compatible.single_client.framework.function_util import (
    api_oneflow_function as global_function,
)
from oneflow.compatible.single_client.framework.generator import (
    create_generator as Generator,
)
from oneflow.compatible.single_client.framework.generator import manual_seed
from oneflow.compatible.single_client.framework.input_blob_def import (
    DeprecatedFixedTensorDef as FixedTensorDef,
)
from oneflow.compatible.single_client.framework.input_blob_def import (
    DeprecatedMirroredTensorDef as MirroredTensorDef,
)
from oneflow.compatible.single_client.framework.job_set_util import (
    inter_job_reuse_mem_strategy,
)
from oneflow.compatible.single_client.framework.model import Model
from oneflow.compatible.single_client.framework.ops import api_acc as acc
from oneflow.compatible.single_client.framework.ops import (
    api_hierarchical_parallel_cast as hierarchical_parallel_cast,
)
from oneflow.compatible.single_client.framework.ops import api_pack as pack
from oneflow.compatible.single_client.framework.ops import (
    api_parallel_cast as parallel_cast,
)
from oneflow.compatible.single_client.framework.ops import api_repeat as repeat
from oneflow.compatible.single_client.framework.ops import api_unpack as unpack
from oneflow.compatible.single_client.framework.placement_util import (
    deprecated_placement as device_prior_placement,
)
from oneflow.compatible.single_client.framework.placement_util import (
    deprecated_placement as fixed_placement,
)
from oneflow.compatible.single_client.framework.scope_util import (
    api_current_scope as current_scope,
)
from oneflow.compatible.single_client.framework.session_util import (
    TmpInitEagerGlobalSession as InitEagerGlobalSession,
)
from oneflow.compatible.single_client.framework.session_util import (
    api_clear_default_session as clear_default_session,
)
from oneflow.compatible.single_client.framework.session_util import (
    api_eager_execution_enabled as eager_execution_enabled,
)
from oneflow.compatible.single_client.framework.session_util import (
    api_find_or_create_module as find_or_create_module,
)
from oneflow.compatible.single_client.framework.session_util import (
    api_sync_default_session as sync_default_session,
)
from oneflow.compatible.single_client.framework.tensor import Tensor
from oneflow.compatible.single_client.ops.array_ops import amp_white_identity
from oneflow.compatible.single_client.ops.array_ops import (
    api_slice_update as slice_update,
)
from oneflow.compatible.single_client.ops.array_ops import (
    argwhere,
    broadcast_like,
    cast_to_static_shape,
    concat,
    dim_gather,
    dynamic_reshape,
    elem_cnt,
    expand,
    expand_dims,
    flatten,
    gather,
    gather_nd,
    identity,
    identity_n,
    masked_fill,
    nonzero,
    ones,
    reshape,
    reshape_like,
    reverse,
    scatter_nd,
    slice,
    slice_v2,
    squeeze,
    stack,
    sync_dynamic_resize,
    tensor_scatter_nd_add,
    tensor_scatter_nd_update,
    transpose,
    where,
    zeros,
)
from oneflow.compatible.single_client.ops.assign_op import assign
from oneflow.compatible.single_client.ops.stateful_ops import StatefulOp as stateful_op
from oneflow.compatible.single_client.ops.categorical_ordinal_encode_op import (
    categorical_ordinal_encode,
)
from oneflow.compatible.single_client.ops.combined_margin_loss import (
    combined_margin_loss,
)
from oneflow.compatible.single_client.ops.constant_op import (
    constant,
    constant_like,
    constant_scalar,
    ones_like,
    zeros_like,
)
from oneflow.compatible.single_client.ops.count_not_finite import (
    count_not_finite,
    multi_count_not_finite,
)
from oneflow.compatible.single_client.ops.diag_ops import diag
from oneflow.compatible.single_client.ops.eager_nccl_ops import eager_nccl_all_reduce
from oneflow.compatible.single_client.ops.get_variable import (
    api_get_variable as get_variable,
)
from oneflow.compatible.single_client.ops.initializer_util import (
    constant_initializer,
    empty_initializer,
)
from oneflow.compatible.single_client.ops.initializer_util import (
    glorot_normal_initializer,
)
from oneflow.compatible.single_client.ops.initializer_util import (
    glorot_normal_initializer as xavier_normal_initializer,
)
from oneflow.compatible.single_client.ops.initializer_util import (
    glorot_uniform_initializer,
)
from oneflow.compatible.single_client.ops.initializer_util import (
    glorot_uniform_initializer as xavier_uniform_initializer,
)
from oneflow.compatible.single_client.ops.initializer_util import (
    kaiming_initializer,
    ones_initializer,
    random_normal_initializer,
    random_uniform_initializer,
    truncated_normal_initializer,
    variance_scaling_initializer,
    zeros_initializer,
)
from oneflow.compatible.single_client.ops.linalg import matmul
from oneflow.compatible.single_client.ops.loss_ops import ctc_loss, smooth_l1_loss
from oneflow.compatible.single_client.ops.math_ops import (
    broadcast_to_compatible_with as broadcast_to_compatible_with,
)
from oneflow.compatible.single_client.ops.math_ops import cast
from oneflow.compatible.single_client.ops.math_ops import clip_by_value as clamp
from oneflow.compatible.single_client.ops.math_ops import clip_by_value as clip
from oneflow.compatible.single_client.ops.math_ops import (
    clip_by_value as clip_by_scalar,
)
from oneflow.compatible.single_client.ops.math_ops import clip_by_value as clip_by_value
from oneflow.compatible.single_client.ops.math_ops import in_top_k as in_top_k
from oneflow.compatible.single_client.ops.math_ops import range
from oneflow.compatible.single_client.ops.math_ops import (
    unsorted_batch_segment_sum as unsorted_batch_segment_sum,
)
from oneflow.compatible.single_client.ops.math_ops import (
    unsorted_segment_sum as unsorted_segment_sum,
)
from oneflow.compatible.single_client.ops.math_ops import (
    unsorted_segment_sum_like as unsorted_segment_sum_like,
)
from oneflow.compatible.single_client.ops.one_hot import one_hot
from oneflow.compatible.single_client.ops.pad import (
    constant_pad2d,
    pad,
    pad_grad,
    reflection_pad2d,
    replication_pad2d,
    same_padding,
    zero_pad2d,
)
from oneflow.compatible.single_client.ops.partial_fc_sample import (
    distributed_partial_fc_sample,
)
from oneflow.compatible.single_client.ops.sort_ops import argsort, sort
from oneflow.compatible.single_client.ops.tensor_buffer_ops import (
    gen_tensor_buffer,
    tensor_buffer_to_list_of_tensors,
    tensor_buffer_to_tensor,
    tensor_to_tensor_buffer,
)
from oneflow.compatible.single_client.ops.user_data_ops import (
    api_image_random_crop as image_random_crop,
)
from oneflow.compatible.single_client.ops.user_data_ops import (
    api_image_resize as image_resize,
)
from oneflow.compatible.single_client.ops.user_data_ops import (
    api_image_target_resize as image_target_resize,
)
from oneflow.compatible.single_client.ops.user_data_ops import (
    image_batch_align as image_batch_align,
)
from oneflow.compatible.single_client.ops.user_data_ops import (
    image_decode as image_decode,
)
from oneflow.compatible.single_client.ops.user_data_ops import image_flip as image_flip
from oneflow.compatible.single_client.ops.user_data_ops import (
    image_normalize as image_normalize,
)
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
from oneflow.compatible.single_client.ops.user_op_builder import (
    api_consistent_user_op_builder as consistent_user_op_builder,
)
from oneflow.compatible.single_client.ops.user_op_builder import (
    api_consistent_user_op_module_builder as consistent_user_op_module_builder,
)
from oneflow.compatible.single_client.ops.user_op_builder import (
    api_user_op_builder as user_op_builder,
)
from oneflow.compatible.single_client.ops.user_op_builder import (
    api_user_op_module_builder as user_op_module_builder,
)
from oneflow.compatible.single_client.ops.watch import Watch as watch
from oneflow.compatible.single_client.ops.watch import WatchDiff as watch_diff

from . import (
    checkpoint,
    config,
    data,
    distribute,
    distributed,
    env,
    image,
    layers,
    losses,
    math,
    model,
    optimizer,
    profiler,
    random,
    regularizers,
    saved_model,
    scope,
    summary,
    sysconfig,
    tensorrt,
    train,
    typing,
    util,
)
