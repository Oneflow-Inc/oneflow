from __future__ import absolute_import

from oneflow.python.oneflow_export import oneflow_export

@oneflow_export("parallel.split")
def split(axis):
    assert type(axis) is int
    assert str(axis) in _axis_str2split_axis_obj, "not a valid split. expected: [0, 11)"
    return _axis_str2split_axis_obj[str(axis)]

@oneflow_export("parallel.broadcast")
def broadcast():
    return _broadcast

@oneflow_export("parallel.auto")
def auto():
    return _auto

@oneflow_export("parallel.assert_is_valid_parallel")
def assert_is_valid_parallel(parallel):
    assert isinstance(parallel, Parallel), \
        '''not a valid parallel policy. 
           expected: 1) oneflow.parallel.split(axis); 2) oneflow.parallel.broadcast(); 3) oneflow.parallel.auto()'''
    
class Parallel(object):
    def __init__(self):
        pass

class AutoParallel(Parallel):
    def __init__(self):
        Parallel.__init__(self)

class BroadcastParallel(Parallel):
    def __init__(self):
        Parallel.__init__(self)

class SplitParallel(Parallel):
    def __init__(self, axis):
        Parallel.__init__(self)
        self.axis_ = axis

    @property
    def axis(self): return self.axis_

_auto = AutoParallel()
_broadcast = BroadcastParallel()
_axis_str2split_axis_obj = dict()
for i in range(11):
    class_name = "Split_Axis%d" % i
    _axis_str2split_axis_obj[str(i)] = SplitParallel(i)
