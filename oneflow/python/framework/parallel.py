from __future__ import absolute_import

from oneflow.python.oneflow_export import oneflow_export

@oneflow_export("parallel.split_axis")
def split_axis(axis):
    assert type(axis) is int
    assert str(axis) in _axis_str2split_axis_obj, "not valid split_axis. expected: [0, 11)"
    return _axis_str2split_axis_obj[str(axis)]

@oneflow_export("parallel.broadcast")
def broadcast():
    return _broadcast

@oneflow_export("parallel.auto")
def auto():
    return _auto

@oneflow_export("parallel.assert_valid_parallel")
def assert_valid_parallel(parallel):
    assert isinstance(parallel, ParallelPolicy), \
        '''not a valid parallel policy. 
           expected: 1) oneflow.split_axis(axis); 2) oneflow.broadcast(); 3) oneflow.auto()'''
    
class ParallelPolicy(object):
    pass

_auto = type("Auto", (ParallelPolicy,), dict(__str__ = lambda self: "Auto"))()
_broadcast = type("Broadcast", (ParallelPolicy,), dict(__str__ = lambda self: "Broadcast"))()
_axis_str2split_axis_obj = dict()
for i in range(11):
    class_name = "SplitAxis_%d" % i
    _axis_str2split_axis_obj[str(i)] = \
        type(class_name, (ParallelPolicy,), dict(__str__ = lambda self: class_name))()
