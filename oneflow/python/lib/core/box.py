from __future__ import absolute_import

from oneflow.python.oneflow_export import oneflow_export

@oneflow_export('Box')
class Box(object):
    def __init__(self):
        self.value_ = None
        self.has_value_ = False

    @property
    def value(self):
        assert self.has_value_
        return self.value_

    @property
    def value_setter(self):
        return lambda val: self.set_value(val)

    def set_value(self, val):
        self.value_ = val
        self.has_value_ = True 

    def has_value(self):
        return self.has_value_
