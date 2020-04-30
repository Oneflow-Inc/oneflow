from __future__ import absolute_import

import oneflow.python.framework.id_util as id_util
from oneflow.python.oneflow_export import oneflow_export

import oneflow as flow

@oneflow_export("math.is_non_decreasing")
def is_non_decreasing(input, name=None):
    x = flow.slice_v2(input,[(1, None, None)])
    y = flow.slice_v2(input,[(None,-1,None)])
    return flow.math.reduce_all(flow.math.greater_equal(x,y))

@oneflow_export("math.is_strictly_increasing")
def is_strictly_increasing(input, name=None):
    x = flow.slice_v2(input,[(1, None, None)])
    y = flow.slice_v2(input,[(None,-1,None)])
    return flow.math.reduce_all(flow.math.greater(x,y))

def _Diff(input):
    return None