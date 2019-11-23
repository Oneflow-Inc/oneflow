from __future__ import absolute_import

import oneflow.python.framework.placement_context as placement_context
import oneflow.python.framework.job_builder as job_builder

from contextlib import contextmanager
from oneflow.python.oneflow_export import oneflow_export
from oneflow.python.framework.variable_getter_composite import (
    VariableGetterComposite,
)


@oneflow_export("experimental.variable_getter")
def variable_getter(fn):
    global cur_job_variable_getter_composite
    if cur_job_variable_getter_composite is None:
        cur_job_variable_getter_composite = VariableGetterComposite()

    cur_job_variable_getter_composite.register(fn)


@contextmanager
def CurJobConf(job_conf):
    _ResetCurJobConf(job_conf)
    yield None
    _ResetCurJobConf(None)


class BeforeNonInputOpBuildAndInferHook:
    def __init__(self, hook):
        self.hook_ = hook

    def __enter__(self):
        before_non_input_op_build_and_infer_hooks.append(self.hook_)

    def __exit__(self, *arg):
        global before_non_input_op_build_and_infer_hooks
        before_non_input_op_build_and_infer_hooks = []


def CurJobAddOp(op_conf, parallel_conf=None):
    return _CurJobAddNonInputOp(op_conf, parallel_conf)


def CurJobAddInputOp(op_conf):
    op_conf.device_type = placement_context.CurPlacementGroupGetDeviceType(
        op_conf
    )

    job_builder.CurCtxAddAndInferOp(
        op_conf, placement_context.ParallelConf4OpConf(op_conf)
    )


def _CurJobAddNonInputOp(op_conf, parallel_conf=None):
    _prefixing_op_name_if_need(op_conf)
    op_conf.device_type = placement_context.CurPlacementGroupGetDeviceType(op_conf)
    for callback in before_non_input_op_build_and_infer_hooks: callback()
    if parallel_conf is None: parallel_conf = placement_context.ParallelConf4OpConf(op_conf)
    job_builder.CurCtxAddAndInferOp(op_conf, parallel_conf)


def _ResetCurJobConf(job_conf):
    global cur_job_conf
    cur_job_conf = job_conf

    global cur_job_var_op_name2var_blob
    cur_job_var_op_name2var_blob = {}

    global cur_job_variable_scope_stack
    assert len(cur_job_variable_scope_stack) == 0
    cur_job_variable_scope_stack = []

    global cur_job_variable_getter_composite
    cur_job_variable_getter_composite = None

    _init_variable_getter()

    global before_non_input_op_build_and_infer_hooks
    before_non_input_op_build_and_infer_hooks = []

    global cur_job_distribution_name_scope
    assert cur_job_distribution_name_scope == ""
    cur_job_distribution_name_scope = ""

    global cur_job_distribution_name_scope_exclude_variable
    cur_job_distribution_name_scope_exclude_variable = True

    global cur_job_distribution_strategy
    cur_job_distribution_strategy = None

def _init_variable_getter():
    @variable_getter
    def get_shared_variable(gen_op_fn, name, *arg, **kwargs):
        global cur_job_var_op_name2var_blob
        if name in cur_job_var_op_name2var_blob:
            return cur_job_var_op_name2var_blob[name]
        else:
            cur_job_var_op_name2var_blob[name] = gen_op_fn()
            return cur_job_var_op_name2var_blob[name]

def _prefixing_op_name_if_need(op_conf):
    if op_conf.HasField("variable_conf"):
        return

    if op_conf.HasField("decode_ofrecord_conf"):
        return

    if op_conf.HasField("layer_norm_conf"):
        pass

    op_conf.name = (
        _get_distribution_prefix(op_conf)
        + _get_variable_prefix()
        + op_conf.name
    )

def _get_variable_prefix():
    global cur_job_variable_scope_stack
    if len(cur_job_variable_scope_stack) == 0:
        return ""

    return "-".join(cur_job_variable_scope_stack) + "-"


def _get_distribution_prefix(op_conf):
    global cur_job_distribution_name_scope
    global cur_job_distribution_name_scope_exclude_variable

    if cur_job_distribution_name_scope_exclude_variable and op_conf.HasField(
        "variable_conf"
    ):
        return ""

    return cur_job_distribution_name_scope


cur_job_conf = None
cur_job_var_op_name2var_blob = {}
before_non_input_op_build_and_infer_hooks = []
cur_job_variable_scope_stack = []
cur_job_variable_getter_composite = None
cur_job_distribution_name_scope = ""
cur_job_distribution_name_scope_exclude_variable = True
cur_job_distribution_strategy = None
