from __future__ import absolute_import

import oneflow.python.framework.compile_context as compile_context
import oneflow.python.framework.remote_blob as remote_blob_util
import oneflow.python.framework.distribute as distribute_util
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util

from oneflow.python.oneflow_export import oneflow_export
import os

@oneflow_export("get_variable")
def get_variable(
    name,
    shape=None,
    dtype=None,
    initializer=None,
    trainable=None,
    model_name=None,
    random_seed=None,
    distribute=distribute_util.broadcast(),
):
    assert isinstance(name, str)
    name = compile_context.GetVariablePrefix() + name

    assert shape is not None, "Argument shape should not be None when the variable exists!"

    op_conf = op_conf_util.OperatorConf()
    op_conf.name = name
    op_conf.variable_conf.shape.dim.extend(shape)

    if dtype is not None:
        op_conf.variable_conf.data_type = dtype
    root_path = compile_context.GetCurJobConfigProto().default_initialize_with_snapshot_path
    dir_path = os.path.join(root_path, name)
    file_path = os.path.join(dir_path, "out")
    if root_path and os.path.isfile(file_path):
        op_conf.variable_conf.initialize_with_snapshot.path = dir_path
        op_conf.variable_conf.initialize_with_snapshot.key = "out"
    else:
        if root_path:
            print("{} not found, will be initialized".format(file_path))
        if initializer is not None:
            op_conf.variable_conf.initializer.CopyFrom(initializer)

    if trainable is not None:
        op_conf.trainable = trainable

    if model_name is not None:
        op_conf.variable_conf.model_name = model_name

    if type(distribute) is distribute_util.SplitDistribute:
        op_conf.variable_conf.split_axis.value = distribute.axis
    else:
        op_conf.variable_conf.split_axis.ClearField("value")

    if random_seed is not None:
        op_conf.variable_conf.random_seed = random_seed

    op_conf.variable_conf.out = "out"

    op_conf, parallel_conf = compile_context.GetOpConfAndParallelConf(op_conf)
    return CreateVariableBlob(op_conf, parallel_conf)

def CreateVariableBlob(op_conf, parallel_conf):
    compile_context.CurJobAddConsistentOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = op_conf.variable_conf.out
    return remote_blob_util.RemoteBlob(lbi)
