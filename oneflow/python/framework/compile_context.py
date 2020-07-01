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


def CurJobAddOp(op_conf, parallel_conf=None):
    # TODO: tsai: remove this debug code when transition ends
    import os

    if (
        os.getenv("ENABLE_USER_OP") != "False"
        and op_conf.HasField("user_conf") == False
    ):
        op_type = op_conf.WhichOneof("op_type")
        if op_type not in logged_op_confs and op_type != "return_conf":
            print("non-user op added: {}".format(op_type))
            logged_op_confs.add(op_type)
    if distribute_ctx.IsMirroredStrategyEnabled():
        return CurJobAddMirroredOp(op_conf, parallel_conf)
    return CurJobAddConsistentOp(op_conf, parallel_conf)


def CurJobAddConsistentOp(op_conf, parallel_conf=None):
    op_conf, parallel_conf = GetOpConfAndParallelConf(op_conf, parallel_conf)
    return c_api_util.CurJobBuildAndInferCtx_AddAndInferConsistentOp(
        op_conf, parallel_conf
    )


def CurJobAddMirroredOp(op_conf, parallel_conf=None):
    assert not hob.consistent_view_enabled(None)
    op_conf, parallel_conf = GetOpConfAndParallelConf(op_conf, parallel_conf)
    return c_api_util.CurJobBuildAndInferCtx_AddAndInferMirroredOp(
        op_conf, parallel_conf
    )


def GetOpConfAndParallelConf(op_conf, parallel_conf=None):
    name_scope.PrependOpNamePrefixIfNeed(op_conf)
    if not op_conf.HasField("device_type"):
        op_conf.device_type = placement_context.CurPlacementGroupGetDeviceType(op_conf)
    if parallel_conf is None:
        parallel_conf = placement_context.ParallelConf4OpConf(op_conf)
    return op_conf, parallel_conf
