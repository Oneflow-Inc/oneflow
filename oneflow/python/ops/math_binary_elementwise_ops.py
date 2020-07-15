from __future__ import absolute_import

import os

import oneflow as flow
import oneflow.python.framework.id_util as id_util
from oneflow.python.oneflow_export import oneflow_export


def build_math_binary_elementwise_op(math_op, x, y, name=None):
    if name is None:
        name = id_util.UniqueStr(math_op + "_")
    return (
        flow.user_op_builder(name)
        .Op(math_op)
        .Input("x", [x])
        .Input("y", [y])
        .Output("z")
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


@oneflow_export("math.atan2")
def atan2(x, y, name=None):
    return build_math_binary_elementwise_op("atan2", x, y, name)


@oneflow_export("math.pow")
def pow(x, y, name=None):
    return build_math_binary_elementwise_op("pow", x, y, name)


@oneflow_export("math.floordiv")
def floordiv(x, y, name=None):
    return build_math_binary_elementwise_op("floordiv", x, y, name)


@oneflow_export("math.xdivy")
def xdivy(x, y, name=None):
    return build_math_binary_elementwise_op("xdivy", x, y, name)


@oneflow_export("math.xlogy")
def xlogy(x, y, name=None):
    return build_math_binary_elementwise_op("xlogy", x, y, name)
