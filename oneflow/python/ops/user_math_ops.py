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

@oneflow_export("math.atan")
def atan(x, name=None):
    if name is None:
        name = id_util.UniqueStr("Atan_")
    return user_op_builder.UserOpConfWrapperBuilder(name).Op("unary")\
            .Input("x",[x])\
            .Output("y")\
            .SetAttr("unary_math_type", "Atan", "AttrTypeString")\
            .Build().RemoteBlobList()[0]

@oneflow_export("math.atanh")
def atanh(x, name=None):
    if name is None:
        name = id_util.UniqueStr("Atanh_")
    return user_op_builder.UserOpConfWrapperBuilder(name).Op("unary")\
            .Input("x",[x])\
            .Output("y")\
            .SetAttr("unary_math_type", "Atanh", "AttrTypeString")\
            .Build().RemoteBlobList()[0]

@oneflow_export("math.ceil")
def ceil(x, name=None):
    if name is None:
        name = id_util.UniqueStr("Ceil_")
    return user_op_builder.UserOpConfWrapperBuilder(name).Op("unary")\
            .Input("x",[x])\
            .Output("y")\
            .SetAttr("unary_math_type", "Ceil", "AttrTypeString")\
            .Build().RemoteBlobList()[0]

@oneflow_export("math.cos")
def cos(x, name=None):
    if name is None:
        name = id_util.UniqueStr("Cos_")
    return user_op_builder.UserOpConfWrapperBuilder(name).Op("unary")\
            .Input("x",[x])\
            .Output("y")\
            .SetAttr("unary_math_type", "Cos", "AttrTypeString")\
            .Build().RemoteBlobList()[0]

@oneflow_export("math.cosh")
def cosh(x, name=None):
    if name is None:
        name = id_util.UniqueStr("Cosh_")
    return user_op_builder.UserOpConfWrapperBuilder(name).Op("unary")\
            .Input("x",[x])\
            .Output("y")\
            .SetAttr("unary_math_type", "Cosh", "AttrTypeString")\
            .Build().RemoteBlobList()[0]

@oneflow_export("math.erf")
def erf(x, name=None):
    if name is None:
        name = id_util.UniqueStr("Erf_")
    return user_op_builder.UserOpConfWrapperBuilder(name).Op("unary")\
            .Input("x",[x])\
            .Output("y")\
            .SetAttr("unary_math_type", "Erf", "AttrTypeString")\
            .Build().RemoteBlobList()[0]

@oneflow_export("math.erfc")
def erfc(x, name=None):
    if name is None:
        name = id_util.UniqueStr("Erfc_")
    return user_op_builder.UserOpConfWrapperBuilder(name).Op("unary")\
            .Input("x",[x])\
            .Output("y")\
            .SetAttr("unary_math_type", "Erfc", "AttrTypeString")\
            .Build().RemoteBlobList()[0]

@oneflow_export("math.exp")
def exp(x, name=None):
    if name is None:
        name = id_util.UniqueStr("Exp_")
    return user_op_builder.UserOpConfWrapperBuilder(name).Op("unary")\
            .Input("x",[x])\
            .Output("y")\
            .SetAttr("unary_math_type", "Exp", "AttrTypeString")\
            .Build().RemoteBlobList()[0]

@oneflow_export("math.expm1")
def expm1(x, name=None):
    if name is None:
        name = id_util.UniqueStr("Expm1_")
    return user_op_builder.UserOpConfWrapperBuilder(name).Op("unary")\
            .Input("x",[x])\
            .Output("y")\
            .SetAttr("unary_math_type", "Expm1", "AttrTypeString")\
            .Build().RemoteBlobList()[0]

@oneflow_export("math.floor")
def floor(x, name=None):
    if name is None:
        name = id_util.UniqueStr("Floor_")
    return user_op_builder.UserOpConfWrapperBuilder(name).Op("unary")\
            .Input("x",[x])\
            .Output("y")\
            .SetAttr("unary_math_type", "Floor", "AttrTypeString")\
            .Build().RemoteBlobList()[0]

@oneflow_export("math.lgamma")
def lgamma(x, name=None):
    if name is None:
        name = id_util.UniqueStr("Lgamma_")
    return user_op_builder.UserOpConfWrapperBuilder(name).Op("unary")\
            .Input("x",[x])\
            .Output("y")\
            .SetAttr("unary_math_type", "Lgamma", "AttrTypeString")\
            .Build().RemoteBlobList()[0]

@oneflow_export("math.log")
def log(x, name=None):
    if name is None:
        name = id_util.UniqueStr("Log_")
    return user_op_builder.UserOpConfWrapperBuilder(name).Op("unary")\
            .Input("x",[x])\
            .Output("y")\
            .SetAttr("unary_math_type", "Log", "AttrTypeString")\
            .Build().RemoteBlobList()[0]

@oneflow_export("math.log1p")
def log1p(x, name=None):
    if name is None:
        name = id_util.UniqueStr("Log1p_")
    return user_op_builder.UserOpConfWrapperBuilder(name).Op("unary")\
            .Input("x",[x])\
            .Output("y")\
            .SetAttr("unary_math_type", "Log1p", "AttrTypeString")\
            .Build().RemoteBlobList()[0]

@oneflow_export("math.log_sigmoid")
def log_sigmoid(x, name=None):
    if name is None:
        name = id_util.UniqueStr("LogSigmoid_")
    return user_op_builder.UserOpConfWrapperBuilder(name).Op("unary")\
            .Input("x",[x])\
            .Output("y")\
            .SetAttr("unary_math_type", "LogSigmoid", "AttrTypeString")\
            .Build().RemoteBlobList()[0]

@oneflow_export("math.negative")
def negative(x, name=None):
    if name is None:
        name = id_util.UniqueStr("Negative_")
    return user_op_builder.UserOpConfWrapperBuilder(name).Op("unary")\
            .Input("x",[x])\
            .Output("y")\
            .SetAttr("unary_math_type", "Negative", "AttrTypeString")\
            .Build().RemoteBlobList()[0]

