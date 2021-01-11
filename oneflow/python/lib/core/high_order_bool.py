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
import oneflow
import oneflow_api


def bool_functor(verbose_debug_str):
    def Decorator(match_function):
        return HighOrderBool(verbose_debug_str, match_function)

    return Decorator


def hob_context_attr(attr_name):
    def Decorator(attr_getter):
        return HobContextAttr(attr_name, attr_getter)

    return Decorator


class BoolFunctor(object):
    def debug_str(self, ctx, display_result=True):
        if hasattr(self, "__debug_str__"):
            if display_result:
                return '"%s"[%s]' % (self.__debug_str__, self(ctx))
            else:
                return '"%s"' % self.__debug_str__
        return self.verbose_debug_str(ctx, display_result=display_result)

    def verbose_debug_str(self, ctx, display_result=True):
        raise NotImplementedError

    def __call__(self, ctx):
        raise NotImplementedError

    def __and__(self, rhs):
        return _AndBoolFunctor(self, rhs)

    def __or__(self, rhs):
        return _OrBoolFunctor(self, rhs)

    def __invert__(self):
        return _NotBoolFunctor(self)


class HighOrderBool(BoolFunctor):
    def __init__(self, verbose_debug_str, function):
        self.verbose_debug_str_ = verbose_debug_str
        self.function_ = function

    def verbose_debug_str(self, ctx, display_result=True):
        if display_result:
            return '"%s"[%s]' % (self.verbose_debug_str_, self.function_(ctx))
        else:
            return '"%s"' % self.verbose_debug_str_

    def __call__(self, ctx):
        return self.function_(ctx)


always_true = HighOrderBool("Always true", lambda: True)
always_false = HighOrderBool("Always false", lambda: False)


class _AndBoolFunctor(BoolFunctor):
    def __init__(self, lhs, rhs):
        assert isinstance(lhs, BoolFunctor)
        assert isinstance(rhs, BoolFunctor)
        self.lhs_ = lhs
        self.rhs_ = rhs

    def verbose_debug_str(self, ctx, display_result=True):
        left_display = self.lhs_.debug_str(ctx, display_result)
        display_result = display_result and self.lhs_(ctx)
        right_display = self.rhs_.debug_str(ctx, display_result)
        return "(%s and %s)" % (left_display, right_display)

    def __call__(self, ctx):
        return self.lhs_(ctx) and self.rhs_(ctx)


class _OrBoolFunctor(BoolFunctor):
    def __init__(self, lhs, rhs):
        assert isinstance(lhs, BoolFunctor)
        assert isinstance(rhs, BoolFunctor)
        self.lhs_ = lhs
        self.rhs_ = rhs

    def verbose_debug_str(self, ctx, display_result=True):
        left_display = self.lhs_.debug_str(ctx, display_result)
        display_result = display_result and (not self.lhs_(ctx))
        right_display = self.rhs_.debug_str(ctx, display_result)
        return "(%s or %s)" % (left_display, right_display)

    def __call__(self, ctx):
        return self.lhs_(ctx) or self.rhs_(ctx)


class _NotBoolFunctor(BoolFunctor):
    def __init__(self, x):
        assert isinstance(x, BoolFunctor)
        self.x_ = x

    def verbose_debug_str(self, ctx, display_result=True):
        return "(not %s)" % self.x_.debug_str(ctx, display_result)

    def __call__(self, ctx):
        return not self.x_(ctx)


class HobContextGetter(object):
    def __init__(self, attr_name, attr_getter):
        self.attr_name_ = attr_name
        self.attr_getter_ = attr_getter

    @property
    def attr_name(self):
        return self.attr_name_

    @property
    def attr_getter(self):
        return self.attr_getter_

    def __eq__(self, other):
        if not isinstance(other, HobContextGetter):
            other = HobContextConstant(other)
        return self._MakeHob(other, "==", lambda a, b: a == b)

    def __ne__(self, other):
        if not isinstance(other, HobContextGetter):
            other = HobContextConstant(other)
        return self._MakeHob(other, "!=", lambda a, b: a != b)

    def __gt__(self, other):
        if not isinstance(other, HobContextGetter):
            other = HobContextConstant(other)
        return self._MakeHob(other, ">", lambda a, b: a > b)

    def __ge__(self, other):
        if not isinstance(other, HobContextGetter):
            other = HobContextConstant(other)
        return self._MakeHob(other, ">=", lambda a, b: a >= b)

    def __lt__(self, other):
        if not isinstance(other, HobContextGetter):
            other = HobContextConstant(other)
        return self._MakeHob(other, "<", lambda a, b: a < b)

    def __le__(self, other):
        if not isinstance(other, HobContextGetter):
            other = HobContextConstant(other)
        return self._MakeHob(other, "<=", lambda a, b: a <= b)

    def _MakeHob(self, other, cmp_str, cmp_func):
        @bool_functor("%s %s %s" % (self.attr_name, cmp_str, other.attr_name))
        def HobHob(context):
            return cmp_func(self.attr_getter(context), other.attr_getter(context))

        return HobHob


class HobContextConstant(HobContextGetter):
    def __init__(self, value):
        HobContextGetter.__init__(self, str(value), lambda ctx: value)


class HobContextAttr(HobContextGetter):
    def __init__(self, attr_name, attr_getter):
        HobContextGetter.__init__(self, attr_name, attr_getter)

    def __getattr__(self, attr_name):
        @hob_context_attr("%s.%s" % (self.attr_name, attr_name))
        def HobCtxAttr(ctx):
            obj = self.attr_getter(ctx)
            if isinstance(obj, oneflow_api.CfgMessage):
                return getattr(obj, attr_name)()
            else:
                return getattr(obj, attr_name)

        return HobCtxAttr

    def HasField(self, attr_name):
        @bool_functor('%s.HasField("%s")' % (self.attr_name, attr_name))
        def BoolFunctor(ctx):
            obj = self.attr_getter(ctx)
            if isinstance(obj, oneflow_api.CfgMessage):
                assert hasattr(obj, "has_" + attr_name), type(obj)
                return getattr(obj, "has_" + attr_name)()
            elif hasattr(obj, "HasField"):
                return obj.HasField(attr_name)
            else:
                return hasattr(obj, attr_name)

        return BoolFunctor
