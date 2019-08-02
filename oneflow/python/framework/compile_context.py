from __future__ import absolute_import

import oneflow.python.framework.placement_context as placement_context

cur_job = None
op_name2variable_blob = {}

def ResetCurJob(job):
    global cur_job
    cur_job = job
    global op_name2variable_blob
    op_name2variable_blob = {}

def CurJobAddOp(op_conf):
    cur_job.net.op.add().CopyFrom(op_conf)
    placement_context.CurPlacementGroupAddOpConf(op_conf)
