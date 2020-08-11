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
from __future__ import absolute_import

from contextlib import contextmanager

import oneflow
import oneflow.python.experimental.name_scope as name_scope
import oneflow.python.framework.c_api_util as c_api_util
import oneflow.python.framework.distribute_context as distribute_ctx
import oneflow.python.framework.placement_context as placement_context
import oneflow.python.framework.session_context as session_ctx
import oneflow.python.framework.hob as hob
import oneflow.python.lib.core.enable_if as enable_if
import oneflow.python.experimental.name_scope as name_scope
import oneflow


def GetCurJobConfigProto():
    return enable_if.unique([GetEagerCurJobConfigProto, GetLazyCurJobConfigProto])()


@enable_if.condition(hob.in_global_mode & hob.eager_execution_enabled)
def GetEagerCurJobConfigProto():
    function_desc = session_ctx.GetDefaultSession().CurrentEagerGlobalFunctionDesc()
    assert function_desc is not None
    return function_desc.job_config_proto


@enable_if.condition(hob.in_global_mode & ~hob.eager_execution_enabled)
def GetLazyCurJobConfigProto():
    job_name = c_api_util.JobBuildAndInferCtx_GetCurrentJobName()
    function_desc = session_ctx.GetDefaultSession().GetLazyFunctionDesc(job_name)
    assert function_desc is not None
    return function_desc.job_config_proto


logged_op_confs = set({})


def CurJobAddOp(op_conf, scope_symbol=None):
    if distribute_ctx.IsMirroredStrategyEnabled():
        return CurJobAddMirroredOp(op_conf, scope_symbol)
    return CurJobAddConsistentOp(op_conf, scope_symbol)


def CurJobAddConsistentOp(op_conf, scope_symbol=None):
    if scope_symbol is None:
        scope_symbol = oneflow.current_scope()
    op_conf.scope_symbol_id = scope_symbol.symbol_id
    if not op_conf.HasField("device_tag"):
        device_tag = scope_symbol.device_parallel_desc_symbol.device_tag
        op_conf.device_tag = device_tag
    return c_api_util.CurJobBuildAndInferCtx_AddAndInferConsistentOp(op_conf)


def CurJobAddMirroredOp(op_conf, scope_symbol=None):
    assert not hob.consistent_view_enabled(None)
    if scope_symbol is None:
        scope_symbol = oneflow.current_scope()
    op_conf.scope_symbol_id = scope_symbol.symbol_id
    if not op_conf.HasField("device_tag"):
        device_tag = scope_symbol.device_parallel_desc_symbol.device_tag
        op_conf.device_tag = device_tag
    return c_api_util.CurJobBuildAndInferCtx_AddAndInferMirroredOp(op_conf)
