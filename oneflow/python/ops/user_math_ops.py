from __future__ import absolute_import

import oneflow.python.framework.compile_context as compile_context
import oneflow.python.framework.remote_blob as remote_blob_util
import oneflow.python.framework.id_util as id_util
import oneflow.python.ops.user_op_builder as user_op_builder
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util

from oneflow.python.oneflow_export import oneflow_export

@oneflow_export("math.abs")
def abs(x, name=None):
    if name is None:
        name = id_util.UniqueStr("Abs_")
    return user_op_builder.UserOpConfWrapperBuilder(name).Op("unary")\
            .Input("x",[x])\
            .Output("y")\
            .SetAttr("unary_math_type", "Abs", "AttrTypeString")\
            .Build().RemoteBlobList()[0]

@oneflow_export("math.acos")
def acos(x, name=None):
    if name is None:
        name = id_util.UniqueStr("Acos_")
    return user_op_builder.UserOpConfWrapperBuilder(name).Op("unary")\
            .Input("x",[x])\
            .Output("y")\
            .SetAttr("unary_math_type", "Acos", "AttrTypeString")\
            .Build().RemoteBlobList()[0]

@oneflow_export("math.acosh")
def acosh(x, name=None):
    if name is None:
        name = id_util.UniqueStr("Acosh_")
    return user_op_builder.UserOpConfWrapperBuilder(name).Op("unary")\
            .Input("x",[x])\
            .Output("y")\
            .SetAttr("unary_math_type", "Acosh", "AttrTypeString")\
            .Build().RemoteBlobList()[0]

@oneflow_export("math.asin")
def asin(x, name=None):
    if name is None:
        name = id_util.UniqueStr("Asin_")
    return user_op_builder.UserOpConfWrapperBuilder(name).Op("unary")\
            .Input("x",[x])\
            .Output("y")\
            .SetAttr("unary_math_type", "Asin", "AttrTypeString")\
            .Build().RemoteBlobList()[0]

@oneflow_export("math.asinh")
def asinh(x, name=None):
    if name is None:
        name = id_util.UniqueStr("Asinh_")
    return user_op_builder.UserOpConfWrapperBuilder(name).Op("unary")\
            .Input("x",[x])\
            .Output("y")\
            .SetAttr("unary_math_type", "Asinh", "AttrTypeString")\
            .Build().RemoteBlobList()[0]


