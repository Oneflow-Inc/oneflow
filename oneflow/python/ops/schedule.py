from __future__ import absolute_import

import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util
import oneflow.python.framework.id_util as id_util
import oneflow.python.framework.compile_context as compile_context
import oneflow.python.framework.remote_blob as remote_blob_util
from oneflow.python.oneflow_export import oneflow_export


@oneflow_export("schedule")
def schedule(train_step, model_update_conf, learning_rate, name=None):
    op_conf = op_conf_util.OperatorConf()
    setattr(op_conf, "name", name if name is not None else id_util.UniqueStr("Scheduler_"))
    setattr(op_conf.learning_rate_schedule_conf, "train_step", train_step.logical_blob_name)
    setattr(op_conf.learning_rate_schedule_conf, "learning_rate", learning_rate)
    if model_update_conf.HasField("learning_rate_decay"):
        op_conf.learning_rate_schedule_conf.learning_rate_decay.CopyFrom(model_update_conf.learning_rate_decay)
    if model_update_conf.HasField("warmup_conf"):
        op_conf.learning_rate_schedule_conf.warmup_conf.CopyFrom(model_update_conf.warmup_conf)
    setattr(op_conf.learning_rate_schedule_conf, "out", "out")
    compile_context.CurJobAddOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)
