from __future__ import absolute_import

from contextlib import contextmanager
import oneflow.python.framework.placement_context as placement_context
import oneflow.python.framework.c_api_util as c_api_util
import oneflow.python.framework.distribute_context as distribute_ctx
import oneflow.python.framework.session_context as session_ctx

def GetCurJobConfigProto():
    job_name = c_api_util.JobBuildAndInferCtx_GetCurrentJobName()
    return session_ctx.GetDefaultSession().GetJobConfigProto(job_name)

def CurJobAddOp(op_conf, parallel_conf=None):
    if distribute_ctx.IsMirroredStrategyEnabled(): return CurJobAddMirroredOp(op_conf, parallel_conf)
    return CurJobAddConsistentOp(op_conf, parallel_conf)

def CurJobAddConsistentOp(op_conf, parallel_conf=None):
    op_conf, parallel_conf = _GetOpConfAndParallelConf(op_conf, parallel_conf)
    return c_api_util.CurJobBuildAndInferCtx_AddAndInferConsistentOp(op_conf, parallel_conf)

def CurJobAddMirroredOp(op_conf, parallel_conf=None):
    op_conf, parallel_conf = _GetOpConfAndParallelConf(op_conf, parallel_conf)
    return c_api_util.CurJobBuildAndInferCtx_AddAndInferMirroredOp(op_conf, parallel_conf)

def ResetCurJobContext():
    global cur_job_var_op_name2var_blob
    cur_job_var_op_name2var_blob = {}

    global cur_job_variable_scope_stack
    assert len(cur_job_variable_scope_stack) == 0
    cur_job_variable_scope_stack = []

def _GetOpConfAndParallelConf(op_conf, parallel_conf):
    _PrependOpNamePrefixIfNeed(op_conf)
    op_conf.device_type = placement_context.CurPlacementGroupGetDeviceType(op_conf)
    if parallel_conf is None: parallel_conf = placement_context.ParallelConf4OpConf(op_conf)
    return op_conf, parallel_conf

def _PrependOpNamePrefixIfNeed(op_conf):
    if op_conf.HasField("variable_conf"):
        return

    if op_conf.HasField("decode_ofrecord_conf"):
        return

    if op_conf.HasField("layer_norm_conf"):
        pass

    op_conf.name = GetVariablePrefix() + op_conf.name

def GetVariablePrefix():
    global cur_job_variable_scope_stack
    if len(cur_job_variable_scope_stack) == 0:
        return ""

    return "-".join(cur_job_variable_scope_stack) + "-"

cur_job_var_op_name2var_blob = {}
cur_job_variable_scope_stack = []
