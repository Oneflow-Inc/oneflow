from __future__ import absolute_import

import oneflow.python.framework.session_context as session_ctx


class DistributeStrategy(object):
    pass


def PushMirroredStrategyEnabled(val):
    session_ctx.GetDefaultSession().is_mirrored_strategy_enabled_stack.append(val)


def IsMirroredStrategyEnabled():
    stack = session_ctx.GetDefaultSession().is_mirrored_strategy_enabled_stack
    return len(stack) > 0 and stack[-1]


def IsConsistentStrategyEnabled():
    stack = session_ctx.GetDefaultSession().is_mirrored_strategy_enabled_stack
    return len(stack) > 0 and not stack[-1]


def PopMirroredStrategyEnabled():
    session_ctx.GetDefaultSession().is_mirrored_strategy_enabled_stack.pop()
