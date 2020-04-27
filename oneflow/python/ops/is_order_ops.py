from __future__ import absolute_import

import oneflow.python.framework.id_util as id_util
from oneflow.python.oneflow_export import oneflow_export

import oneflow as flow

@oneflow_export("is_non_decreasing")
def is_non_decreasing(input, name=None):
    order_type = "NON_DECREASING"
    unique_str = "IsNonDecreasing_"
    return (_IsOrderOp(input, order_type, unique_str, name))

@oneflow_export("is_strictly_increasing")
def is_strictly_increasing(input, name=None):
    order_type = "STRICTLY_INCREASING"
    unique_str = "IsStrictlyIncreasing_"
    return (_IsOrderOp(input, order_type, unique_str, name))

def _IsOrderOp(input, order_type, unique_str, name=None):
    assert order_type in ["NON_DECREASING", "STRICTLY_INCREASING"]
    assert unique_str in ["IsNonDecreasing_", "IsStrictlyIncreasing_"]
    return (
        flow.user_op_builder(name if name is not None else id_util.UniqueStr(unique_str))
        .Op("is_order")
        .Input("in", [input])
        .Output("out")
        .SetAttr("order_type", order_type, "AttrTypeString")
        .Build()
        .RemoteBlobList()[0]
    )