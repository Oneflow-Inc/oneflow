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
import oneflow.core.job.scope_pb2 as scope_pb2_util
import oneflow.core.job.job_conf_pb2 as job_conf_pb
import oneflow.framework.attr_util as attr_util
import oneflow.framework.c_api_util as c_api_util
import oneflow.framework.scope_util as scope_util
import oneflow.framework.session_context as session_context
from oneflow.framework.tensor import Tensor
from oneflow.nn.graph.proxy import GraphBlockType
import oneflow._oneflow_internal._C as _C

lazy_mode = oneflow._oneflow_internal.lazy_mode


@contextmanager
def graph_build_context(config_proto, session):
    prev_scope = oneflow._oneflow_internal.GetCurrentScope()
    assert type(config_proto) is job_conf_pb.JobConfigProto, type(config_proto)
    config_proto_str = text_format.MessageToString(config_proto)
    new_scope = oneflow._oneflow_internal.MakeInitialScope(
        config_proto_str, oneflow.placement("cpu", [0]), False,  # is_local
    )

    graph_scope = _make_new_graph_scope(new_scope, config_proto.job_name)

    oneflow._oneflow_internal.eager.Sync()
    with lazy_mode.guard(True):
        with JobBuildAndInferCtx(config_proto):
            with BlockScopeContext(prev_scope, graph_scope):
                yield


class JobBuildAndInferCtx(object):
    def __init__(self, config_proto):
        self._job_conf = config_proto

    def __enter__(self):
        c_api_util.JobBuildAndInferCtx_Open(self._job_conf.job_name)
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


class DebugScopeContext(object):
    def __init__(
        self,
        s_level,
        v_level=0,
        mode=False,
        max_py_stack_depth=2,
        only_user_py_stack=True,
    ):
        self._prev_v = oneflow._oneflow_internal.GetFLAGS_v()
        self._prev_logtostderr = oneflow._oneflow_internal.GetFLAGS_alsologtostderr()
        self._prev_mode = oneflow._oneflow_internal.GetGraphDebugMode()
        self._prev_max_py_stack_depth = (
            oneflow._oneflow_internal.GetGraphDebugMaxPyStackDepth()
        )
        self._prev_only_user_py_stack = (
            oneflow._oneflow_internal.GetGraphDebugOnlyUserPyStack()
        )
        self._v = max(v_level, self._prev_v)
        self._mode = mode
        self._s = s_level
        self._max_py_stack_depth = max(
            max_py_stack_depth, self._prev_max_py_stack_depth
        )
        self._only_user_py_stack = only_user_py_stack

    def __enter__(self):
        oneflow._oneflow_internal.SetFLAGS_v(self._v)
        oneflow._oneflow_internal.SetGraphDebugMode(self._mode)
        if self._s == 0 and self._v >= 1:
            oneflow._oneflow_internal.SetFLAGS_alsologtostderr(True)
        oneflow._oneflow_internal.SetGraphDebugMaxPyStackDepth(self._max_py_stack_depth)
        oneflow._oneflow_internal.SetGraphDebugOnlyUserPyStack(self._only_user_py_stack)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._s == 0 and self._v >= 1:
            oneflow._oneflow_internal.SetFLAGS_alsologtostderr(self._prev_logtostderr)
        oneflow._oneflow_internal.SetFLAGS_v(self._prev_v)
        oneflow._oneflow_internal.SetGraphDebugMode(self._prev_mode)
        oneflow._oneflow_internal.SetGraphDebugMaxPyStackDepth(
            self._prev_max_py_stack_depth
        )
        oneflow._oneflow_internal.SetGraphDebugOnlyUserPyStack(
            self._prev_only_user_py_stack
        )


def _make_new_scope(prev_scope, scope_proto_str_setter):
    new_scope = None

    def build_scope(builder):
        nonlocal new_scope
        new_scope = builder.BuildScopeByProtoStrSetter(
            prev_scope, scope_proto_str_setter
        )
        assert new_scope is not None

    oneflow._oneflow_internal.deprecated.PhysicalRun(build_scope)
    oneflow._oneflow_internal.eager.Sync()
    return new_scope


def _make_new_graph_scope(prev_scope, graph_name):
    assert prev_scope is not None
    attr_dict = dict()
    name2default = session_context.GetDefaultSession().scope_attr_name2default_val

    def scope_proto_str_setter(serialized_scope_proto: str):
        scope_proto = text_format.Parse(
            serialized_scope_proto, scope_pb2_util.ScopeProto()
        )
        scope_proto.module_name = graph_name
        return str(text_format.MessageToString(scope_proto))

    return _make_new_scope(prev_scope, scope_proto_str_setter)


def make_new_blockgraph_scope(prev_scope, graph_block):
    assert prev_scope is not None
    assert graph_block is not None
    attr_dict = dict()
    if graph_block.stage_id is not None:
        attr_dict["pipeline_stage_id_hint"] = graph_block.stage_id
    if graph_block.type == GraphBlockType.MODULE:
        if graph_block.activation_checkpointing is not None:
            attr_dict["checkpointing"] = graph_block.activation_checkpointing

    name2default = session_context.GetDefaultSession().scope_attr_name2default_val

    def scope_proto_str_setter(serialized_scope_proto: str):
        scope_proto = text_format.Parse(
            serialized_scope_proto, scope_pb2_util.ScopeProto()
        )
        # set attr
        for attr_name, py_value in attr_dict.items():
            assert attr_name in name2default
            attr_util.SetProtoAttrValue(
                scope_proto.attr_name2attr_value[attr_name],
                py_value,
                name2default[attr_name],
            )
        # append name prefix
        scope_proto.ClearField("scope_op_name_prefixes")
        scope_proto.scope_op_name_prefixes.append(
            graph_block.name_prefix + graph_block.name
        )
        # set module name
        if graph_block.type == GraphBlockType.MODULE:
            scope_proto.module_name = graph_block.name_prefix + graph_block.name
        return str(text_format.MessageToString(scope_proto))

    return _make_new_scope(prev_scope, scope_proto_str_setter)


def make_new_name_scope(prev_scope, name):
    assert prev_scope is not None

    def scope_proto_str_setter(serialized_scope_proto: str):
        scope_proto = text_format.Parse(
            serialized_scope_proto, scope_pb2_util.ScopeProto()
        )
        # append name prefix
        scope_proto.ClearField("scope_op_name_prefixes")
        scope_proto.scope_op_name_prefixes.append(name)
        scope_proto.module_name = name
        return str(text_format.MessageToString(scope_proto))

    return _make_new_scope(prev_scope, scope_proto_str_setter)


def scope_to_proto(scope):
    return text_format.Parse(scope._proto_str, scope_pb2_util.ScopeProto())


def build_graph_input_arg(op_name, arg):
    assert isinstance(arg, Tensor)
    input_conf = oneflow.core.operator.op_conf_pb2.FeedInputOpConf()
    input_conf.in_0 = "in_0"  # Set the default value, otherwise the parsing fails
    input_conf.out_0 = "out_0"
    input_conf_str = text_format.MessageToString(input_conf)

    input_op = oneflow._oneflow_internal.one.FeedInputOpExpr(
        op_name, input_conf_str, ["in_0"], ["out_0"]
    )
    lazy_arg = _C.dispatch_feed_input(input_op, arg)
    return lazy_arg


def build_graph_state(op_name, state_tensor, state_config):
    var_conf = oneflow.core.operator.op_conf_pb2.FeedVariableOpConf()
    var_conf.in_0 = "in_0"  # Set the default value, otherwise the parsing fails
    var_conf.out_0 = "out_0"
    var_conf_str = text_format.MessageToString(var_conf)

    var_op = oneflow._oneflow_internal.one.FeedVariableOpExpr(
        op_name, var_conf_str, ["in_0"], ["out_0"]
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

    output_conf = oneflow.core.operator.op_conf_pb2.FetchOutputOpConf()
    output_conf.in_0 = "in_0"  # Set the default value, otherwise the parsing fails
    output_conf.out_0 = "out_0"
    output_conf_str = text_format.MessageToString(output_conf)

    output_op = oneflow._oneflow_internal.one.FetchOutputOpExpr(
        op_name, output_conf_str, ["in_0"], ["out_0"]
    )
    fake_eager_out = _C.dispatch_fetch_output(output_op, out)
    return fake_eager_out
