from __future__ import absolute_import

import oneflow.python.framework.id_util as id_util
from oneflow.python.oneflow_export import oneflow_export

import oneflow as flow

@oneflow_export("is_non_decreasing")
def is_non_decreasing(input, name=None):
    order_type = "NON_DECREASING"
    return (
        flow.user_op_builder(name if name is not None else id_util.UniqueStr("IsNonDecreasing_"))
        .Op("is_order")
        .Input("in", [input])
        .Output("out")
        .SetAttr("order_type", order_type, "AttrTypeString")
        .Build()
        .RemoteBlobList()[0]
    )

@oneflow_export("is_strictly_increasing")
def is_strictly_increasing(input, name=None):
    order_type = "STRICTLY_INCREASING"
    return (
        flow.user_op_builder(name if name is not None else id_util.UniqueStr("IsStrictlyIncreasing_"))
        .Op("is_order")
        .Input("in", [input])
        .Output("out")
        .SetAttr("order_type", order_type, "AttrTypeString")
        .Build()
        .RemoteBlobList()[0]
    )