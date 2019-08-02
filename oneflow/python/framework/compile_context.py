from __future__ import absolute_import

import oneflow.python.framework.placement_context as placement_context

cur_job = None
cur_job_var_op_name2var_blob = {}

def ResetCurJob(job):
    global cur_job
    cur_job = job
    global cur_job_var_op_name2var_blob 
    cur_job_var_op_name2var_blob = {}

def CurJobAddOp(op_conf):
    cur_job.net.op.add().CopyFrom(op_conf)
    placement_context.CurPlacementGroupAddOpConf(op_conf)
