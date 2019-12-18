from __future__ import absolute_import

import uuid
import oneflow.python.framework.compile_context as compile_context
import oneflow.python.framework.id_util as id_util
import oneflow.python.framework.watcher as watcher_util
import oneflow.core.operator.op_conf_pb2 as op_conf_util
from oneflow.core.job.lbi_diff_watcher_info_pb2 import LbiAndDiffWatcherUuidPair
from oneflow.python.framework.remote_blob import MirroredBlob, ConsistentBlob
import oneflow.python.framework.local_blob as local_blob_util
import oneflow.python.framework.c_api_util as c_api_util

from oneflow.python.oneflow_export import oneflow_export

@oneflow_export("watch")
def Watch(blob_watched, handler_or_prompt=None):
    handler = _MakeHandler(handler_or_prompt)
    if type(blob_watched) is ConsistentBlob:
        handler_uuid = str(uuid.uuid1())
        op_conf = op_conf_util.OperatorConf()
        op_conf.name = id_util.UniqueStr("ForeignWatch_")
        setattr(op_conf.foreign_watch_conf, "in", blob_watched.logical_blob_name)
        op_conf.foreign_watch_conf.handler_uuid = handler_uuid
        compile_context.CurJobAddOp(op_conf, blob_watched.parallel_conf)
        watcher_util.BindUuidAndHandler(handler_uuid, blob_watched, handler)
    elif type(blob_watched) is MirroredBlob:
        handlers = _MakeSubConsistentBlobHandlers(blob_watched, handler)
        for consistent_blob, sub_handler in zip(blob_watched.sub_consistent_blob_list, handlers):
            assert type(consistent_blob) is ConsistentBlob
            Watch(consistent_blob, sub_handler)
    else:
        raise NotImplementedError

@oneflow_export("watch_diff")
def WatchDiff(blob_watched, handler_or_prompt=None):
    handler = _MakeHandler(handler_or_prompt)
    if type(blob_watched) is ConsistentBlob:
        handler_uuid = str(uuid.uuid1())
        lbi_and_uuid = LbiAndDiffWatcherUuidPair()
        lbi_and_uuid.lbi.CopyFrom(blob_watched.lbi)
        lbi_and_uuid.watcher_uuid = handler_uuid
        c_api_util.CurJobBuildAndInferCtx_AddLbiAndDiffWatcherUuidPair(lbi_and_uuid)
        watcher_util.BindUuidAndHandler(handler_uuid, blob_watched, handler)
    elif type(blob_watched) is MirroredBlob:
        handlers = _MakeSubConsistentBlobHandlers(blob_watched, handler)
        for consistent_blob, sub_handler in zip(blob_watched.sub_consistent_blob_list, handlers):
            assert type(consistent_blob) is ConsistentBlob
            WatchDiff(consistent_blob, sub_handler)
    else:
        raise NotImplementedError

def _MakeHandler(handler_or_prompt):
    if callable(handler_or_prompt): return handler_or_prompt
    prompt = handler_or_prompt
    def Handler(x):
        if prompt is not None: print(str(prompt))
        print(x)
    return Handler

def _MakeSubConsistentBlobHandlers(blob_watched, handler):
    assert type(blob_watched) is MirroredBlob
    handler4parallel_id_and_local_blob = _MakeHandler4ParallelIdAndLocalBlob(blob_watched, handler)
    return [_WrapperHandler4ParallelIdAndLocalBlob(i, handler4parallel_id_and_local_blob)
            for i in range(len(blob_watched.sub_consistent_blob_list))]

def _WrapperHandler4ParallelIdAndLocalBlob(parallel_id, handler4parallel_id_and_local_blob):
    return lambda local_blob: handler4parallel_id_and_local_blob(parallel_id, local_blob)

def _MakeHandler4ParallelIdAndLocalBlob(blob_watched, handler):
    parallel_id2consistent_local_blob = {}
    len_sub_remote_blobs = len(blob_watched.sub_consistent_blob_list)
    def HandlerParallelIdAndLocalBlob(parallel_id, local_blob):
        assert parallel_id not in parallel_id2consistent_local_blob
        parallel_id2consistent_local_blob[parallel_id] = local_blob
        if len(parallel_id2consistent_local_blob) != len_sub_remote_blobs: return
        local_blob_list = [parallel_id2consistent_local_blob[parallel_id]
                           for i in range(len_sub_remote_blobs)]
        handler(local_blob_util.MergeLocalBlobs(local_blob_list, blob_watched))
    return HandlerParallelIdAndLocalBlob
