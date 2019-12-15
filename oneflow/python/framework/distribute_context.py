from __future__ import absolute_import

class DistributeStrategy(object):
    pass

def PushMirrorStrategyEnabled(val):
    global _is_mirror_strategy_enabled_stack
    _is_mirror_strategy_enabled_stack.append(val)

def IsMirrorStrategyEnabled():
    assert len(_is_mirror_strategy_enabled_stack) > 0
    return _is_mirror_strategy_enabled_stack[-1]

def AnyStrategyEnabled():
    return len(_is_mirror_strategy_enabled_stack) > 0

def PophMirrorStrategyEnabled():
    global _is_mirror_strategy_enabled_stack
    _is_mirror_strategy_enabled_stack.pop()

_is_mirror_strategy_enabled_stack = []
