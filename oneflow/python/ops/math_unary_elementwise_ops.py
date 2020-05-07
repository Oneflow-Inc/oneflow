from __future__ import absolute_import

import oneflow.python.framework.compile_context as compile_context
import oneflow.python.framework.remote_blob as remote_blob_util
import oneflow.python.framework.id_util as id_util
import oneflow.python.ops.user_op_builder as user_op_builder
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util

from oneflow.python.oneflow_export import oneflow_export

def build_unary_elemwise_math_op(math_op, x, name=None):
    if name is None:
        name = id_util.UniqueStr(math_op + "_")
    return user_op_builder.UserOpConfWrapperBuilder(name).Op("math_unary_elementwise")\
        .Input("x", [x])\
        .Output("y")\
        .SetAttr("math_type", math_op, "AttrTypeString")\
        .Build().RemoteBlobList()[0]

@oneflow_export("math.abs")
def abs(x, name=None):
    return build_unary_elemwise_math_op("Abs", x, name)

@oneflow_export("math.acos")
def acos(x, name=None):
    return build_unary_elemwise_math_op("Acos", x, name)

@oneflow_export("math.acosh")
def acosh(x, name=None):
    return build_unary_elemwise_math_op("Acosh", x, name)

@oneflow_export("math.asin")
def asin(x, name=None):
    return build_unary_elemwise_math_op("Asin", x, name)

@oneflow_export("math.asinh")
def asinh(x, name=None):
    return build_unary_elemwise_math_op("Asinh", x, name)

@oneflow_export("math.atan")
def atan(x, name=None):
    return build_unary_elemwise_math_op("Atan", x, name)

@oneflow_export("math.atanh")
def atanh(x, name=None):
    return build_unary_elemwise_math_op("Atanh", x, name)

@oneflow_export("math.ceil")
def ceil(x, name=None):
    return build_unary_elemwise_math_op("Ceil", x, name)

@oneflow_export("math.cos")
def cos(x, name=None):
    return build_unary_elemwise_math_op("Cos", x, name)

@oneflow_export("math.cosh")
def cosh(x, name=None):
    return build_unary_elemwise_math_op("Cosh", x, name)

@oneflow_export("math.erf")
def erf(x, name=None):
    return build_unary_elemwise_math_op("Erf", x, name)

@oneflow_export("math.erfc")
def erfc(x, name=None):
    return build_unary_elemwise_math_op("Erfc", x, name)

@oneflow_export("math.exp")
def exp(x, name=None):
    return build_unary_elemwise_math_op("Exp", x, name)

@oneflow_export("math.expm1")
def expm1(x, name=None):
    return build_unary_elemwise_math_op("Expm1", x, name)

@oneflow_export("math.floor")
def floor(x, name=None):
    return build_unary_elemwise_math_op("Floor", x, name)

@oneflow_export("math.lgamma")
def lgamma(x, name=None):
    return build_unary_elemwise_math_op("Lgamma", x, name)

@oneflow_export("math.log")
def log(x, name=None):
    return build_unary_elemwise_math_op("Log", x, name)

@oneflow_export("math.log1p")
def log1p(x, name=None):
    return build_unary_elemwise_math_op("Log1p", x, name)

@oneflow_export("math.log_sigmoid")
def log_sigmoid(x, name=None):
    return build_unary_elemwise_math_op("LogSigmoid", x, name)

@oneflow_export("math.negative")
def negative(x, name=None):
    return build_unary_elemwise_math_op("Negative", x, name)

@oneflow_export("math.reciprocal")
def reciprocal(x, name=None):
    return build_unary_elemwise_math_op("Reciprocal", x, name)

@oneflow_export("math.reciprocal_no_nan")
def reciprocal_no_nan(x, name=None):
    return build_unary_elemwise_math_op("ReciprocalNoNan", x, name)

@oneflow_export("math.rint")
def rint(x, name=None):
    return build_unary_elemwise_math_op("Rint", x, name)

@oneflow_export("math.round")
def round(x, name=None):
    return build_unary_elemwise_math_op("Round", x, name)

@oneflow_export("math.rsqrt")
def rsqrt(x, name=None):
    if os.getenv("ENABLE_USER_OP") == 'True':
        return build_unary_elemwise_math_op("Rsqrt", x, name)

    op_conf = op_conf_util.OperatorConf()
    setattr(op_conf, "name", name if name is not None else id_util.UniqueStr("Rsqrt_"))
    setattr(op_conf.rsqrt_conf, "in", x.logical_blob_name)
    setattr(op_conf.rsqrt_conf, "out", "out")
    compile_context.CurJobAddOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)

@oneflow_export("math.sigmoid_v2")
def sigmoid_v2(x, name=None):
    return build_unary_elemwise_math_op("Sigmoid", x, name)

@oneflow_export("math.sign")
def sign(x, name=None):
    return build_unary_elemwise_math_op("Sign", x, name)

@oneflow_export("math.sin")
def sin(x, name=None):
    return build_unary_elemwise_math_op("Sin", x, name)

@oneflow_export("math.sinh")
def sinh(x, name=None):
    return build_unary_elemwise_math_op("Sinh", x, name)

@oneflow_export("math.softplus")
def softplus(x, name=None):
    return build_unary_elemwise_math_op("Softplus", x, name)

@oneflow_export("math.sqrt")
def sqrt(x, name=None):
    if os.getenv("ENABLE_USER_OP") == 'True':
        return build_unary_elemwise_math_op("Sqrt", x, name)

    op_conf = op_conf_util.OperatorConf()
    setattr(op_conf, "name", name if name is not None else id_util.UniqueStr("Sqrt_"))
    setattr(op_conf.sqrt_conf, "in", x.logical_blob_name)
    setattr(op_conf.sqrt_conf, "out", "out")
    compile_context.CurJobAddOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)

@oneflow_export("math.square")
def square(x, name=None):
    if os.getenv("ENABLE_USER_OP") == 'True':
        return build_unary_elemwise_math_op("Square", x, name)
    op_conf = op_conf_util.OperatorConf()
    setattr(
        op_conf,
        "name",
        name if name is not None else id_util.UniqueStr("Square_"),
    )
    setattr(op_conf.square_conf, "in", x.logical_blob_name)
    setattr(op_conf.square_conf, "out", "out")
    compile_context.CurJobAddOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)

@oneflow_export("math.tan")
def tan(x, name=None):
    return build_unary_elemwise_math_op("Tan", x, name)

@oneflow_export("math.tanh_v2")
def tanh_v2(x, name=None):
    return build_unary_elemwise_math_op("Tanh", x, name)

