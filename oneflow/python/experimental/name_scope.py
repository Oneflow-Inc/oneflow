from __future__ import absolute_import

import oneflow.python.framework.session_context as session_context
from oneflow.python.oneflow_export import oneflow_export
from contextlib import contextmanager


NAME_SCOPE_OP_CONF_BLACKLIST = ["variable_conf", "decode_ofrecord_conf"]


@oneflow_export("experimental.name_scope", "deprecated.variable_scope")
@contextmanager
def name_scope(name):
    assert isinstance(name, str)
    name_scope_stack_push(name)
    try:
        yield None
    finally:
        name_scope_stack_pop()


def name_scope_stack_push(name):
    session_context.GetDefaultSession().op_name_scope_stack.append(name)


def name_scope_stack_pop():
    return session_context.GetDefaultSession().op_name_scope_stack.pop()


def name_scope_stack_top():
    sess = session_context.GetDefaultSession()
    if len(sess.op_name_scope_stack) == 0:
        return ""
    return sess.op_name_scope_stack[-1]


def GetNameScopePrefix():
    sess = session_context.GetDefaultSession()
    if len(sess.op_name_scope_stack) == 0:
        return ""
    return "-".join(sess.op_name_scope_stack) + "-"


def PrependNameScopePrefix4OpConf(op_conf):
    for no_prefix_conf in NAME_SCOPE_OP_CONF_BLACKLIST:
        if op_conf.HasField(no_prefix_conf):
            return

    op_conf.name = GetNameScopePrefix() + op_conf.name
