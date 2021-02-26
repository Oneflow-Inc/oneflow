"""
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from __future__ import absolute_import

import oneflow.python.framework.session_context as session_ctx
import oneflow.python.framework.scope_util as scope_util


class DistributeStrategy(object):
    def __init__(self, is_mirrored):
        self.is_mirrored_ = is_mirrored
        self.scope_context_ = None
        sess = session_ctx.GetDefaultSession()
        # bypass the first DistributeStrategy for avoiding None old_scope
        if sess.is_running and len(sess.is_mirrored_strategy_enabled_stack) > 0:

            def BuildScope(old_scope, builder):
                return builder.BuildScopeWithNewIsMirrored(old_scope, is_mirrored)

            self.scope_context_ = scope_util.ScopeContext(
                scope_util.MakeScope(BuildScope)
            )

    def __enter__(self, *argc, **kwarg):
        PushMirroredStrategyEnabled(self.is_mirrored_)
        if self.scope_context_ is not None:
            self.scope_context_.__enter__(*argc, **kwarg)

    def __exit__(self, *argc, **kwarg):
        PopMirroredStrategyEnabled()
        if self.scope_context_ is not None:
            self.scope_context_.__exit__(*argc, **kwarg)


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
