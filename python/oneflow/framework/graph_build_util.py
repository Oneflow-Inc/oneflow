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
from contextlib import contextmanager
import os

from google.protobuf import text_format
import oneflow

import oneflow._oneflow_internal
from oneflow._oneflow_internal.oneflow.core.framework import (
    user_op_attr as user_op_attr_cfg,
)
import oneflow.core.job.scope_pb2 as scope_pb2_util
import oneflow.framework.attr_util as attr_util
import oneflow.framework.c_api_util as c_api_util
import oneflow.framework.scope_util as scope_util
import oneflow.framework.session_context as session_context
from oneflow.framework.tensor import Tensor

import oneflow._oneflow_internal._C as _C

lazy_mode = oneflow._oneflow_internal.lazy_mode


@contextmanager
def graph_build_context(config_proto, session):
    prev_scope = oneflow._oneflow_internal.GetCurrentScope()
    new_scope = scope_util.MakeInitialScope(
        config_proto,
        "cpu",  # NOTE(chengcheng): graph init scope is useless, just set cpu 0:0 for test.
        ["0:0"],
        None,  # TODO(): set hierarchy from user graph config
        False,  # is_mirrored
    )

    with lazy_mode.guard(True):
        with JobBuildAndInferCtx(config_proto):
            with BlockScopeContext(prev_scope, new_scope):
                yield


class JobBuildAndInferCtx(object):
    def __init__(self, config_proto):
        self._job_conf = config_proto

    def __enter__(self):
        c_api_util.JobBuildAndInferCtx_Open(self._job_conf.job_name())
        c_api_util.CurJobBuildAndInferCtx_SetJobConf(self._job_conf)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            oneflow._oneflow_internal.JobBuildAndInferCtx_Close()
            return True
        else:
            oneflow._oneflow_internal.JobBuildAndInferCtx_Close()
            return False


class BlockScopeContext(object):
    def __init__(self, prev_scope, new_scope):
        assert prev_scope is not None
        assert new_scope is not None
        self._prev_scope = prev_scope
        self._new_scope = new_scope

    def __enter__(self):
        assert oneflow._oneflow_internal.GetCurrentScope() is self._prev_scope
        oneflow._oneflow_internal.GlobalScopeStackPush(self._new_scope)

    def __exit__(self, exc_type, exc_val, exc_tb):
        assert oneflow._oneflow_internal.GetCurrentScope() is self._new_scope
        oneflow._oneflow_internal.GlobalScopeStackPop()
        assert oneflow._oneflow_internal.GetCurrentScope() is self._prev_scope
        if exc_type is None:
            return True
        else:
            return False


class GLogScopeContext(object):
    def __init__(self, s_level, v_level=0):
        self._prev_v = oneflow._oneflow_internal.GetFLAGS_v()
        self._prev_logtostderr = oneflow._oneflow_internal.GetFLAGS_alsologtostderr()
        self._v = v_level
        self._s = s_level

    def __enter__(self):
        oneflow._oneflow_internal.SetFLAGS_v(self._v)
        if self._s == 0 and self._v >= 1:
            oneflow._oneflow_internal.SetFLAGS_alsologtostderr(True)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._s == 0 and self._v >= 1:
            oneflow._oneflow_internal.SetFLAGS_alsologtostderr(self._prev_logtostderr)
        oneflow._oneflow_internal.SetFLAGS_v(self._prev_v)


def make_new_block_scope(prev_scope, block):
    assert prev_scope is not None
    assert block is not None

    attr_dict = dict()
    if block.config.stage_id is not None:
        attr_dict["pipeline_stage_id_hint"] = block.config.stage_id
    if block.config.activation_checkpointing is not None:
        attr_dict["checkpointing"] = block.config.activation_checkpointing

    name2default = session_context.GetDefaultSession().scope_attr_name2default_val

    def scope_proto_setter(scope_proto):
        # set attr
        for attr_name, py_value in attr_dict.items():
            assert attr_name in name2default
            attr_util.SetAttrValue(
                scope_proto.mutable_attr_name2attr_value()[attr_name],
                py_value,
                name2default[attr_name],
            )
        # append name prefix
        scope_proto.clear_scope_op_name_prefixes()
        scope_proto.add_scope_op_name_prefixes(block.name_prefix + block.name)

    new_scope = None

    def build_scope(builder):
        nonlocal new_scope
        new_scope = builder.BuildScopeByProtoSetter(prev_scope, scope_proto_setter)
        assert new_scope is not None

    oneflow._oneflow_internal.deprecated.PhysicalRun(build_scope)
    oneflow._oneflow_internal.eager.multi_client.Sync()
    return new_scope


def scope_to_proto(scope):
    return text_format.Parse(scope._proto_str, scope_pb2_util.ScopeProto())


def build_graph_input_arg(op_name, arg):
    assert isinstance(arg, Tensor)
    input_conf = (
        oneflow._oneflow_internal.oneflow.core.operator.op_conf.FeedInputOpConf()
    )

    input_op = oneflow._oneflow_internal.one.FeedInputOpExpr(
        op_name, input_conf, ["in_0"], ["out_0"]
    )
    lazy_arg = _C.dispatch_feed_input(input_op, arg)
    return lazy_arg


def build_graph_state(op_name, state_tensor, state_config):
    var_conf = (
        oneflow._oneflow_internal.oneflow.core.operator.op_conf.FeedVariableOpConf()
    )

    var_op = oneflow._oneflow_internal.one.FeedVariableOpExpr(
        op_name, var_conf, ["in_0"], ["out_0"]
    )
    l2 = 0.0
    if state_config is not None:
        l2 = state_config.l2
    elif state_tensor.requires_grad:
        l2 = 0.0

    assert isinstance(state_tensor, Tensor)
    lazy_tensor = _C.dispatch_feed_variable(var_op, state_tensor, l2=l2)
    return lazy_tensor


def build_graph_output(op_name, out):
    assert isinstance(out, Tensor)

    output_conf = (
        oneflow._oneflow_internal.oneflow.core.operator.op_conf.FetchOutputOpConf()
    )

    output_op = oneflow._oneflow_internal.one.FetchOutputOpExpr(
        op_name, output_conf, ["in_0"], ["out_0"]
    )
    fake_eager_out = _C.dispatch_fetch_output(output_op, out)

    return fake_eager_out
