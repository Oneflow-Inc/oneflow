from __future__ import absolute_import

import oneflow.python.framework.placement_context as placement_context

cur_job = None

job_name2input_remote_blobs = {}

job_name2output_remote_blobs = {}

def CurJobAddOp(op_conf):
    cur_job.net.op.add().CopyFrom(op_conf)
    placement_context.CurPlacementGroupAddOpName(op_conf.name)
