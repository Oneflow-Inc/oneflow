from __future__ import absolute_import

import uuid
import oneflow.python.framework.compile_context as compile_context
import oneflow.python.framework.id_util as id_util
import oneflow.python.framework.watcher as watcher_util
import oneflow.python.framework.job_builder as job_builder
import oneflow.core.operator.op_conf_pb2 as op_conf_util
from oneflow.core.job.lbi_diff_watcher_info_pb2 import LbiAndDiffWatcherUuidPair
from oneflow.python.framework.remote_blob import MirrorBlob, ConsistentBlob
from oneflow.python.oneflow_export import oneflow_export


def make_handler(prompt):
    def handler(x):
        if prompt is not None:
            print(str(prompt))

        print(x)

    return handler


@oneflow_export("watch")
def watch(blob_watched, param=None):
    if isinstance(param, (tuple, list)):
        handlers = [p if callable(p) else make_handler(p) for p in param]
    elif callable(param):
        handlers = [param]
    else:
        handlers = [make_handler(param)]
    assert len(handlers) > 0

    blobs = []
    if isinstance(blob_watched, MirrorBlob):
        blobs = blob_watched.sub_consistent_blob_list
    elif isinstance(blob_watched, ConsistentBlob):
        blobs = [blob_watched]
    else:
        raise NotImplementedError

    for i, blob in enumerate(blobs):
        handler_uuid = str(uuid.uuid1())
        op_conf = op_conf_util.OperatorConf()
        op_conf.name = id_util.UniqueStr("ForeignWatch_")
        setattr(op_conf.foreign_watch_conf, "in", blob.logical_blob_name)
        op_conf.foreign_watch_conf.handler_uuid = handler_uuid
        compile_context.CurJobAddOp(op_conf, blob.parallel_conf)
        handler = handlers[i] if i < len(handlers) else handlers[0]
        watcher_util.BindUuidAndHandler(handler_uuid, handler)


@oneflow_export("watch_diff")
def watch_diff(blob_watched, param=None):
    if isinstance(param, (tuple, list)):
        handlers = [p if callable(p) else make_handler(p) for p in param]
    elif callable(param):
        handlers = [param]
    else:
        handlers = [make_handler(param)]
    assert len(handlers) > 0

    blobs = []
    if isinstance(blob_watched, MirrorBlob):
        blobs = blob_watched.sub_consistent_blob_list
    elif isinstance(blob_watched, ConsistentBlob):
        blobs = [blob_watched]
    else:
        raise NotImplementedError

    for i, blob in enumerate(blobs):
        handler_uuid = str(uuid.uuid1())
        lbi_and_uuid = LbiAndDiffWatcherUuidPair()
        lbi_and_uuid.lbi.CopyFrom(blob.lbi)
        lbi_and_uuid.watcher_uuid = handler_uuid
        job_builder.CurCtxAddLbiAndDiffWatcherUuidPair(lbi_and_uuid)
        handler = handlers[i] if i < len(handlers) else handlers[0]
        watcher_util.BindUuidAndHandler(handler_uuid, handler)
