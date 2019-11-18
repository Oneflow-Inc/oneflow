from __future__ import absolute_import

import uuid
import collections
import oneflow
import oneflow.python.framework.compile_context as compile_context
import oneflow.python.framework.id_util as id_util
import oneflow.python.framework.watcher as watcher_util
import oneflow.python.framework.job_builder as job_builder
import oneflow.core.operator.op_conf_pb2 as op_conf_util
from oneflow.core.job.lbi_diff_watcher_info_pb2 import LbiAndDiffWatcherUuidPair

from oneflow.python.oneflow_export import oneflow_export

@oneflow_export("watch")
def watch(watched, handler = None):
    if callable(handler) == False:
        prompt = handler
        def _print(x):
            if prompt is not None: print(prompt)
            print(x)
        handler = _print
    handler_uuid = str(uuid.uuid1())
    op_conf = op_conf_util.OperatorConf()
    op_conf.name = id_util.UniqueStr("ForeignWatch_")
    setattr(op_conf.foreign_watch_conf, "in", watched.logical_blob_name)
    op_conf.foreign_watch_conf.handler_uuid = handler_uuid
    compile_context.CurJobAddOp(op_conf, watched.parallel_conf)
    watcher_util.BindUuidAndHandler(handler_uuid, handler)

@oneflow_export("watch_diff")
def watch_diff(watched, handler = None):
    if callable(handler) == False:
        prompt = handler
        def _print(x):
            if prompt is not None: print(prompt)
            print(x)
        handler = _print
    handler_uuid = str(uuid.uuid1())
    lbi_and_uuid = LbiAndDiffWatcherUuidPair()
    lbi_and_uuid.lbi.CopyFrom(watched.lbi)
    lbi_and_uuid.watcher_uuid = handler_uuid
    job_builder.CurCtxAddLbiAndDiffWatcherUuidPair(lbi_and_uuid)
    watcher_util.BindUuidAndHandler(handler_uuid, handler)
