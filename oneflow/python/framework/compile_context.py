from __future__ import absolute_import

import oneflow.python.framework.placement_context as placement_context
import oneflow.python.framework.job_builder as job_builder

def ResetCurJob(job):
    global cur_job
    cur_job = job
    global cur_job_var_op_name2var_blob 
    cur_job_var_op_name2var_blob = {}

def CurJobAddOp(op_conf):
    op_conf.device_type = placement_context.CurPlacementGroupGetDeviceType(op_conf)
    if op_conf.HasField("input_conf"):
        _CurJobAddInputOp(op_conf)
    else:
        _CurJobAddNonInputOp(op_conf)
    placement_context.CurPlacementGroupAddOpConf(op_conf)

def _CurJobAddInputOp(op_conf):
    job_builder.CurCtxAddAndInferInputOp(op_conf)
    
def _CurJobAddNonInputOp(op_conf):
    job_builder.CurCtxSetJobConfIfNotSet(cur_job.job_conf)
    job_builder.CurCtxAddAndInferNonInputOp(op_conf)

cur_job = None
cur_job_var_op_name2var_blob = {}
