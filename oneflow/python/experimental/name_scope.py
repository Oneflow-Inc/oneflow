from __future__ import absolute_import

import oneflow.python.framework.session_context as session_context
import oneflow.python.framework.g_func_ctx as g_func_ctx
from oneflow.python.oneflow_export import oneflow_export
from contextlib import contextmanager


@oneflow_export("name_scope", "experimental.name_scope", "deprecated.variable_scope")
@contextmanager
def name_scope(name):
    assert isinstance(name, str)
    name_scope_stack_push(name)
    try:
        yield None
    finally:
        name_scope_stack_pop()


def name_scope_stack_push(name):
    job_name = g_func_ctx.JobBuildAndInferCtx_GetCurrentJobName()
    sess = session_context.GetDefaultSession()
    if job_name not in sess.job_name2name_scope_stack:
        sess.job_name2name_scope_stack[job_name] = []
    sess.job_name2name_scope_stack[job_name].append(name)


def name_scope_stack_pop():
    job_name = g_func_ctx.JobBuildAndInferCtx_GetCurrentJobName()
    sess = session_context.GetDefaultSession()
    assert job_name in sess.job_name2name_scope_stack
    assert len(sess.job_name2name_scope_stack[job_name]) > 0
    return sess.job_name2name_scope_stack[job_name].pop()


def GetJobNameScopePrefix(job_name):
    sess = session_context.GetDefaultSession()
    if job_name not in sess.job_name2name_scope_stack:
        return ""
    if len(sess.job_name2name_scope_stack[job_name]) == 0:
        return ""
    return "-".join(sess.job_name2name_scope_stack[job_name]) + "-"


def PrependOpNamePrefixIfNeed(op_conf):
    if op_conf.HasField("variable_conf"):
        return

    if op_conf.HasField("decode_ofrecord_conf"):
        return

    if op_conf.HasField("user_conf"):
        return

    job_name = g_func_ctx.JobBuildAndInferCtx_GetCurrentJobName()
    op_conf.name = GetJobNameScopePrefix(job_name) + op_conf.name
