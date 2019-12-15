from __future__ import absolute_import

import oneflow.python.framework.placement_context as placement_context
import oneflow.python.framework.job_builder as job_builder
from contextlib import contextmanager

def CurJobAddOp(op_conf, parallel_conf=None):
    _PrependOpNamePrefixIfNeed(op_conf)
    op_conf.device_type = placement_context.CurPlacementGroupGetDeviceType(op_conf)
    if parallel_conf is None: parallel_conf = placement_context.ParallelConf4OpConf(op_conf)
    job_builder.CurCtxAddAndInferOp(op_conf, parallel_conf)

def ResetCurJobContext():
    global cur_job_var_op_name2var_blob
    cur_job_var_op_name2var_blob = {}

    global cur_job_variable_scope_stack
    assert len(cur_job_variable_scope_stack) == 0
    cur_job_variable_scope_stack = []

def _PrependOpNamePrefixIfNeed(op_conf):
    if op_conf.HasField("variable_conf"):
        return

    if op_conf.HasField("decode_ofrecord_conf"):
        return

    if op_conf.HasField("layer_norm_conf"):
        pass

    op_conf.name = _get_variable_prefix() + op_conf.name


def _get_variable_prefix():
    global cur_job_variable_scope_stack
    if len(cur_job_variable_scope_stack) == 0:
        return ""

    return "-".join(cur_job_variable_scope_stack) + "-"

cur_job_var_op_name2var_blob = {}
cur_job_variable_scope_stack = []
