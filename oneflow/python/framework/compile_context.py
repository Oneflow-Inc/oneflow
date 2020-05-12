from __future__ import absolute_import

from contextlib import contextmanager
import oneflow.python.framework.placement_context as placement_context
import oneflow.python.framework.g_func_ctx as g_func_ctx
import oneflow.python.framework.distribute_context as distribute_ctx
import oneflow.python.framework.session_context as session_ctx
import oneflow.python.experimental.name_scope as name_scope
import oneflow

def GetCurJobConfigProto():
    job_name = g_func_ctx.JobBuildAndInferCtx_GetCurrentJobName()
    return session_ctx.GetDefaultSession().GetJobConfigProto(job_name)

logged_op_confs = set({})
def CurJobAddOp(op_conf, parallel_conf=None):
    # TODO: tsai: remove this debug code when transition ends
    import os
    if os.getenv("ENABLE_USER_OP") == 'True' and op_conf.HasField("user_conf") == False:
        op_type = op_conf.WhichOneof("op_type")
        if op_type not in logged_op_confs and op_type != "return_conf":
            print("non-user op added: {}".format(op_type))
            logged_op_confs.add(op_type)
    if distribute_ctx.IsMirroredStrategyEnabled():
        return CurJobAddMirroredOp(op_conf, parallel_conf)
    return CurJobAddConsistentOp(op_conf, parallel_conf)

def CurJobAddConsistentOp(op_conf, parallel_conf=None):
    op_conf, parallel_conf = GetOpConfAndParallelConf(op_conf, parallel_conf)
    return g_func_ctx.CurJobBuildAndInferCtx_AddAndInferConsistentOp(op_conf, parallel_conf)

def CurJobAddMirroredOp(op_conf, parallel_conf=None):
    op_conf, parallel_conf = GetOpConfAndParallelConf(op_conf, parallel_conf)
    return g_func_ctx.CurJobBuildAndInferCtx_AddAndInferMirroredOp(op_conf, parallel_conf)

def GetOpConfAndParallelConf(op_conf, parallel_conf=None):
    name_scope.PrependOpNamePrefixIfNeed(op_conf)
    if not op_conf.HasField('device_type'):
        op_conf.device_type = placement_context.CurPlacementGroupGetDeviceType(op_conf)
    if parallel_conf is None: parallel_conf = placement_context.ParallelConf4OpConf(op_conf)
    return op_conf, parallel_conf
