from __future__ import absolute_import

import oneflow.python.framework.id_util as id_util
from oneflow.python.oneflow_export import oneflow_export

import oneflow as flow

@oneflow_export("sort")
def sort(input, direction="ASCENDING", name=None):
    assert direction in ["ASCENDING", "DESCENDING"]
    return (
        flow.user_op_builder(name if name is not None else id_util.UniqueStr("Sort_"))
        .Op("sort")
        .Input("in", [input])
        .Output("out")
        .SetAttr("direction", direction, "AttrTypeString")
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )

@oneflow_export("argsort")
def argsort(input, direction="ASCENDING", name=None):
    assert direction in ["ASCENDING", "DESCENDING"]
    return (
        flow.user_op_builder(name if name is not None else id_util.UniqueStr("ArgSort_"))
        .Op("arg_sort")
        .Input("in", [input])
        .Output("out")
        .SetAttr("direction", direction, "AttrTypeString")
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )
