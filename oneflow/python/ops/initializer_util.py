from __future__ import absolute_import

import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.common.data_type_pb2 as data_type_conf_util
from oneflow.python.oneflow_export import oneflow_export


@oneflow_export("constant_initializer")
def constant_initializer(value=0, dtype=data_type_conf_util.kFloat):
    initializer = op_conf_util.InitializerConf()
    if dtype in [data_type_conf_util.kFloat, data_type_conf_util.kDouble]:
        setattr(initializer.constant_conf, "value", float(value))
    elif dtype in [
        data_type_conf_util.kInt8,
        data_type_conf_util.kInt32,
        data_type_conf_util.kInt64,
    ]:
        setattr(initializer.constant_int_conf, "value", int(value))
    else:
        raise NotImplementedError("Do not support such data type")

    return initializer


@oneflow_export("range_initializer")
def range_initializer(start=0, stride=0, axis=-1, dtype=data_type_conf_util.kFloat):
    initializer = op_conf_util.InitializerConf()
    if dtype in [data_type_conf_util.kFloat, data_type_conf_util.kDouble]:
        setattr(initializer.range_conf, "start", float(start))
        setattr(initializer.range_conf, "stride", float(stride))
        setattr(initializer.range_conf, "axis", int(axis))
    elif dtype in [
        data_type_conf_util.kInt8,
        data_type_conf_util.kInt32,
        data_type_conf_util.kInt64,
    ]:
        setattr(initializer.int_range_conf, "start", int(start))
        setattr(initializer.int_range_conf, "stride", int(stride))
        setattr(initializer.int_range_conf, "axis", int(axis))
    else:
        raise NotImplementedError("Do not support such data type")

    return initializer


@oneflow_export("random_uniform_initializer")
def random_uniform_initializer(min=0, max=0, dtype=data_type_conf_util.kFloat):
    initializer = op_conf_util.InitializerConf()
    if dtype in [data_type_conf_util.kFloat, data_type_conf_util.kDouble]:
        setattr(initializer.random_uniform_conf, "min", float(min))
        setattr(initializer.random_uniform_conf, "max", float(max))
    elif dtype in [
        data_type_conf_util.kInt8,
        data_type_conf_util.kInt32,
        data_type_conf_util.kInt64,
    ]:
        setattr(initializer.random_uniform_int_conf, "min", int(min))
        setattr(initializer.random_uniform_int_cof, "max", int(max))
    else:
        raise NotImplementedError("Do not support such data type")

    return initializer


@oneflow_export("random_normal_initializer")
def random_normal_initializer(mean=0.0, std=1.0):
    initializer = op_conf_util.InitializerConf()
    setattr(initializer.random_normal_conf, "mean", float(mean))
    setattr(initializer.random_normal_conf, "std", float(std))

    return initializer


@oneflow_export("truncated_normal_initializer")
def truncated_normal_initializer(std=1.0):
    initializer = op_conf_util.InitializerConf()
    setattr(initializer.truncated_normal_conf, "std", float(std))

    return initializer


@oneflow_export("xavier_initializer")
def xavier_initializer(variance_norm):
    initializer = op_conf_util.InitializerConf()
    setattr(initializer.xavier_conf, "variance_norm", variance_norm)

    return initializer


@oneflow_export("msra_initializer")
def msra_initializer(variance_norm):
    initializer = op_conf_util.InitializerConf()
    setattr(initializer.xavier_conf, "variance_norm", variance_norm)

    return initializer
