class BoolFunctor(object):
    def debug_str(self, display_result=True):
        raise NotImplementedError

    def __call__(self):
        raise NotImplementedError

    def __and__(self, rhs):
        return _AndBoolFunctor(self, rhs)

    def __or__(self, rhs):
        return _OrBoolFunctor(self, rhs)

    def __invert__(self):
        return _NotBoolFunctor(self)


class HighOrderBool(BoolFunctor):
    def __init__(self, debug_str, function):
        self.debug_str_ = debug_str
        self.function_ = function

    def debug_str(self, display_result=True):
        if display_result:
            return '"%s"[%s]' % (self.debug_str_, self.function_())
        else:
            return '"%s"' % self.debug_str_

    def __call__(self):
        return self.function_()


always_true = HighOrderBool("Always true", lambda: True)
always_false = HighOrderBool("Always false", lambda: False)


class _AndBoolFunctor(BoolFunctor):
    def __init__(self, lhs, rhs):
        assert isinstance(lhs, BoolFunctor)
        assert isinstance(rhs, BoolFunctor)
        self.lhs_ = lhs
        self.rhs_ = rhs

    def debug_str(self, display_result=True):
        left_display = self.lhs_.debug_str(display_result)
        display_result = display_result and self.lhs_()
        right_display = self.rhs_.debug_str(display_result)
        return "(%s and %s)" % (left_display, right_display)

    def __call__(self):
        return self.lhs_() and self.rhs_()


class _OrBoolFunctor(BoolFunctor):
    def __init__(self, lhs, rhs):
        assert isinstance(lhs, BoolFunctor)
        assert isinstance(rhs, BoolFunctor)
        self.lhs_ = lhs
        self.rhs_ = rhs

    def debug_str(self, display_result=True):
        left_display = self.lhs_.debug_str(display_result)
        display_result = display_result and (not self.lhs_())
        right_display = self.rhs_.debug_str(display_result)
        return "(%s or %s)" % (left_display, right_display)

    def __call__(self):
        return self.lhs_() or self.rhs_()


class _NotBoolFunctor(BoolFunctor):
    def __init__(self, x):
        assert isinstance(x, BoolFunctor)
        self.x_ = x

    def debug_str(self, display_result=True):
        return "(not %s)" % self.x_.debug_str(display_result)

    def __call__(self):
        return not self.x_()
