from __future__ import absolute_import

import oneflow.python.framework.compile_context as compile_context
from oneflow.python.oneflow_export import oneflow_export
from contextlib import contextmanager


@oneflow_export("deprecated.variable_scope")
@contextmanager
def variable_scope(name):
    assert isinstance(name, str)
    var_scope_stack = get_variable_scope_stack()
    var_scope_stack.append(name)
    try:
        yield None
    finally:
        variable_scope_stack_pop()


def get_variable_scope_stack():
    return compile_context.cur_job_variable_scope_stack


def variable_scope_stack_pop():
    compile_context.cur_job_variable_scope_stack = compile_context.cur_job_variable_scope_stack[
        0:-1
    ]


@oneflow_export("experimental.distribution_name_scope")
@contextmanager
def distribution_name_scope(machine, device, exclude_variables=True):
    assert isinstance(machine, int)
    assert isinstance(device, int)
    assert compile_context.cur_job_distribution_name_scope == ""
    compile_context.cur_job_distribution_name_scope = "m{}d{}-".format(machine, device)
    origin_exclude_variables_flag = (
        compile_context.cur_job_distribution_name_scope_exclude_variable
    )
    compile_context.cur_job_distribution_name_scope_exclude_variable = (
        exclude_variables
    )
    try:
        yield compile_context.cur_job_distribution_name_scope
    finally:
        compile_context.cur_job_distribution_name_scope = ""
        compile_context.cur_job_distribution_name_scope_exclude_variable = (
            origin_exclude_variables_flag
        )
