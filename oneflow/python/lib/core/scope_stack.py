from __future__ import absolute_import

from contextlib import contextmanager


class ScopeStack(object):
    def __init__(self, init=[]):
        if not isinstance(init, list):
            init = [init]
        assert isinstance(init, list)
        self.stack_ = init

    def Current(self):
        assert len(self.stack_) > 0
        return self.stack_[0]

    @contextmanager
    def NewScope(self, scope):
        self.stack_.insert(0, scope)
        yield
        self.stack_.pop(0)
