from __future__ import absolute_import

import oneflow.python.framework.compile_context as compile_context
import oneflow.python.framework.remote_blob as remote_blob_util
import oneflow.python.framework.id_util as id_util
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util

from oneflow.python.oneflow_export import oneflow_export


@oneflow_export('keras.initializers.constant_int_init')
def constant_int_init(value=None):
    init_conf = op_conf_util.InitializerConf()
    if value is not None:
        assert type(value) == int, "Arguments type error!"
        init_conf.constant_int_conf.value = value
    return init_conf


@oneflow_export('keras.initializers.constant_init')
def constant_init(value=None):
    init_conf = op_conf_util.InitializerConf()
    if value is not None:
        init_conf.constant_conf.value = value
    return init_conf


@oneflow_export('keras.initializers.random_uniform_int_init')
def random_uniform_int_init(min_val=None, max_val=None):
    init_conf = op_conf_util.InitializerConf()
    if min_val is not None:
        assert type(min_val) == int, "Arguments type error!"
        init_conf.random_uniform_int_conf.min = min_val
    if max_val is not None:
        assert type(max_val) == int, "Arguments type error!"
        init_conf.random_uniform_int_conf.max = max_val
    return init_conf


@oneflow_export('keras.initializers.random_uniform_init')
def random_uniform_init(min_val=None, max_val=None):
    init_conf = op_conf_util.InitializerConf()
    if min_val is not None:
        init_conf.random_uniform_conf.min = min_val
    if max_val is not None:
        init_conf.random_uniform_conf.max = max_val
    return init_conf


@oneflow_export('keras.initializers.random_normal_init')
def random_normal_init(mean=None, std=None):
    init_conf = op_conf_util.InitializerConf()
    if mean is not None:
        init_conf.random_normal_conf.mean = mean
    if std is not None:
        init_conf.random_normal_conf.std = std
    return init_conf

@oneflow_export('keras.initializers.truncated_normal_init')
def truncated_normal_init(std=None):
    init_conf = op_conf_util.InitializerConf()
    if std is not None:
        init_conf.truncated_normal_conf.std = std
    return init_conf


dict_variance_norm = {  'FAN_IN': op_conf_util.kFanIn,
                        'FAN_OUT': op_conf_util.kFanOut,
                        'FAN_AVG': op_conf_util.kAverage}
@oneflow_export('keras.initializer.xavier_init')
def xavier_init(mode):
    allowed_variance_norms = {  'FAN_IN',
                                'FAN_OUT',
                                'FAN_AVG'}
    init_conf = op_conf.InitializerConf()
    if mode not in allowed_variance_norms:
        raise TypeError('Mode argument not understood!')
    else:
        init_conf.xavier_conf.variance_norm = dict_variance_norm[mode]
    return init_conf


@oneflow_export('keras.initializer.msra_init')
def msra_init(mode):
    allowed_variance_norms = {  'FAN_IN',
                                'FAN_OUT',
                                'FAN_AVG'}
    init_conf = op_conf.InitializerConf()
    if mode not in allowed_variance_norms:
        raise TypeError('Mode argument not understood!')
    else:
        init_conf.msra_conf.variance_norm = dict_variance_norm[mode]
    return init_conf


@oneflow_export('keras.initializer.range_init')
def range_init(start=None, stride=None, axis=None):
    init_conf = op_conf.InitializerConf()
    if start is not None:
        init_conf.range_conf.start = start
    if stride is not None:
        init_conf.range_conf.stride = stride
    if axis is not None:
        init_conf.range_conf.axis = axis
    return init_conf


@oneflow_export('keras.initializer.int_range_init')
def int_range_init(start=None, stride=None, axis=None):
    init_conf = op_conf.InitializerConf()
    if start is not None:
        assert type(start) == int, "Arguments type error!"
        init_conf.int_range_conf.start = start
    if stride is not None:
        assert type(stride) == int, "Arguments type error!"
        init_conf.int_range_conf.stride = stride
    if axis is not None:
        assert type(axis) == int, "Arguments type error!"
        init_conf.int_range_conf.axis = axis
    return init_conf


