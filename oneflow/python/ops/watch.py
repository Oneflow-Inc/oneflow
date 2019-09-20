from __future__ import absolute_import

import uuid
import collections
import oneflow
import oneflow.python.framework.compile_context as compile_context
import oneflow.python.framework.id_util as id_util
import oneflow.python.framework.watcher as watcher_util
import oneflow.core.operator.op_conf_pb2 as op_conf_util

from oneflow.python.oneflow_export import oneflow_export

@oneflow_export("watch")
def watch(watched, handler):
    assert callable(handler)
    watched = list(watched) if isinstance(watched, collections.Sized) else [watched]
    watched_lbn = [x.logical_blob_name for x in watched]
    handler_uuid = str(uuid.uuid1())
    op_conf = op_conf_util.OperatorConf()
    op_conf.name = id_util.UniqueStr("ForeignWatch_")
    getattr(op_conf.foreign_watch_conf, "in").extend(watched_lbn)
    op_conf.foreign_watch_conf.handler_uuid = handler_uuid
    with oneflow.fixed_placement("cpu", "0:0"): compile_context.CurJobAddOp(op_conf)
    watcher_util.BindUuidAndHandler(handler_uuid, handler)
