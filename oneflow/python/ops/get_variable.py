from __future__ import absolute_import

import oneflow.python.framework.compile_context as compile_context
import oneflow.python.framework.remote_blob as remote_blob_util
import oneflow.python.framework.id_util as id_util
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util
import oneflow.core.common.data_type_pb2 as data_type_conf_util

from oneflow.python.oneflow_export import oneflow_export


@oneflow_export('get_variable')
def get_variable(name,
                shape=None,
                dtype=None,
                initializer=None,
                trainable=None,
                model_name=None,
                split_axis=None):
    
    if name not in compile_context.cur_job_var_op_name2var_blob:
        op_conf = op_conf_util.OperatorConf()
        op_conf.name = name
        assert shape is not None, "Argument shape should not be None when the variable exists!"
        op_conf.variable_conf.shape.dim.extend(shape)
        if dtype is not None:
            op_conf.variable_conf.data_type = dtype
        if initializer is not None:
            op_conf.variable_conf.initializer.CopyFrom(initializer)
        if trainable is not None:
            op_conf.trainable = trainable
        if model_name is not None:
            op_conf.variable_conf.model_name = model_name
        if type(split_axis) is int:
            op_conf.variable_conf.split_axis.value = split_axis
        else:
            assert split_axis is None or split_axis is False
            op_conf.variable_conf.split_axis.ClearField("value")
        op_conf.variable_conf.out = "out"
        compile_context.CurJobAddOp(op_conf)
        lbi = logical_blob_id_util.LogicalBlobId()
        lbi.op_name = op_conf.name
        lbi.blob_name = op_conf.variable_conf.out
        var_blob = remote_blob_util.RemoteBlob(lbi)
        compile_context.cur_job_var_op_name2var_blob[name] = var_blob
    return compile_context.cur_job_var_op_name2var_blob[name] 

