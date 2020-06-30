from __future__ import absolute_import

from contextlib import contextmanager

import oneflow.python.framework.c_api_util as c_api_util
import oneflow.python.framework.session_context as session_context
from oneflow.python.oneflow_export import oneflow_export


@oneflow_export("name_scope", "experimental.name_scope", "deprecated.variable_scope")
@contextmanager
def name_scope(name):
    r"""Create a name scope. All variables within the name scope will have a prefix `[SCOPE NAME]-`. This is for convenience only and has no other effect on the system. 
    Usage::

        with oneflow.name_scope("scope1"):
            ...
            with oneflow.name_scope("scope12"):
                ...

    Args:
        name: Name of this name scope

    """
    assert isinstance(name, str)
    name_scope_stack_push(name)
    try:
        yield None
    finally:
        name_scope_stack_pop()


def name_scope_stack_push(name):
    job_name = c_api_util.JobBuildAndInferCtx_GetCurrentJobName()
    sess = session_context.GetDefaultSession()
    if job_name not in sess.job_name2name_scope_stack:
        sess.job_name2name_scope_stack[job_name] = []
    sess.job_name2name_scope_stack[job_name].append(name)


def name_scope_stack_pop():
    job_name = c_api_util.JobBuildAndInferCtx_GetCurrentJobName()
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

    job_name = c_api_util.JobBuildAndInferCtx_GetCurrentJobName()
    op_conf.name = GetJobNameScopePrefix(job_name) + op_conf.name
