from contextlib import contextmanager
from google.protobuf import text_format
import oneflow.core.job.scope_pb2 as scope_pb2_util
import oneflow.framework.attr_util as attr_util
import oneflow.framework.c_api_util as c_api_util
import oneflow.framework.placement_util as placement_util
import oneflow.framework.scope_util as scope_util
import oneflow.framework.session_context as session_context
import oneflow._oneflow_internal
from oneflow._oneflow_internal import Tensor as InternalTensor
from oneflow.framework.tensor import Tensor

lazy_mode = oneflow._oneflow_internal.lazy_mode


@contextmanager
def graph_build_context(config_proto, session):
    prev_scope = oneflow._oneflow_internal.GetCurrentScope()
    new_scope = scope_util.MakeInitialScope(config_proto, "cpu", ["0:0"], None, False)
    with lazy_mode.gard(True):
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
        oneflow._oneflow_internal.JobBuildAndInferCtx_Close()
        if exc_type is None:
            return True
        else:
            return False


class BlockScopeContext(object):
    def __init__(self, prev_scope, new_scope):
        assert prev_scope is not None
        assert new_scope is not None
        self._prev_scope = prev_scope
        self._new_scope = new_scope

    def __enter__(self):
        oneflow._oneflow_internal.GlobalScopeStackPush(self._new_scope)

    def __exit__(self, exc_type, exc_val, exc_tb):
        assert oneflow._oneflow_internal.GetCurrentScope() is self._new_scope
        oneflow._oneflow_internal.GlobalScopeStackPop()
        assert oneflow._oneflow_internal.GetCurrentScope() is self._prev_scope
        if exc_type is None:
            return True
        else:
            return False


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
        for (attr_name, py_value) in attr_dict.items():
            assert attr_name in name2default
            attr_util.SetAttrValue(
                scope_proto.mutable_attr_name2attr_value()[attr_name],
                py_value,
                name2default[attr_name],
            )
        scope_proto.clear_scope_op_name_prefixes()
        scope_proto.add_scope_op_name_prefixes(block.name_prefix + block.name)

    new_scope = None

    def build_scope(builder):
        nonlocal new_scope
        new_scope = builder.BuildScopeByProtoSetter(prev_scope, scope_proto_setter)
        assert new_scope is not None

    oneflow._oneflow_internal.deprecated.LogicalRun(build_scope)
    return new_scope


def scope_to_proto(scope):
    return text_format.Parse(scope._proto_str, scope_pb2_util.ScopeProto())


def build_graph_input_arg(op_name, arg):
    assert isinstance(arg, (Tensor, InternalTensor))
    input_conf = (
        oneflow._oneflow_internal.oneflow.core.operator.op_conf.FeedInputOpConf()
    )
    input_op = oneflow._oneflow_internal.one.FeedInputOpExpr(
        op_name, input_conf, ["in_0"], ["out_0"]
    )
    attrs = oneflow._oneflow_internal.MutableCfgAttrMap()
    if isinstance(arg, Tensor):
        if not arg.is_determined:
            arg.determine()
        tensor_in_c = arg._local_or_consistent_tensor
    else:
        tensor_in_c = arg
    lazy_arg = input_op.apply([tensor_in_c], attrs)[0]
    return lazy_arg


def build_graph_state(op_name, state_tensor):
    var_conf = (
        oneflow._oneflow_internal.oneflow.core.operator.op_conf.FeedVariableOpConf()
    )
    var_op = oneflow._oneflow_internal.one.FeedVariableOpExpr(
        op_name, var_conf, ["in_0"], ["out_0"]
    )
    attrs = oneflow._oneflow_internal.MutableCfgAttrMap()
    assert isinstance(state_tensor, Tensor)
    if not state_tensor.is_determined:
        state_tensor.determine()
    tensor_in_c = state_tensor._local_or_consistent_tensor
    lazy_tensor = var_op.apply([tensor_in_c], attrs)[0]
    return lazy_tensor


def build_graph_output(op_name, out):
    assert isinstance(out, InternalTensor)
    assert out.is_lazy
    output_conf = (
        oneflow._oneflow_internal.oneflow.core.operator.op_conf.FetchOutputOpConf()
    )
    output_op = oneflow._oneflow_internal.one.FetchOutputOpExpr(
        op_name, output_conf, ["in_0"], ["out_0"]
    )
    attrs = oneflow._oneflow_internal.MutableCfgAttrMap()
    eager_out = output_op.apply([out], attrs)[0]
    return eager_out
