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
