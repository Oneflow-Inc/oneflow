from __future__ import absolute_import

from google.protobuf import text_format
import oneflow.core.record.record_pb2 as record_util
import oneflow.python.framework.ofblob as ofblob
import oneflow.oneflow_internal as oneflow_internal
import oneflow.python.framework.c_api_util as c_api_util
import oneflow.python.framework.remote_blob as remote_blob_util
import oneflow.python.framework.local_blob as local_blob_util
import oneflow.python.framework.session_context as session_ctx
import traceback

def BindUuidAndHandler(uuid, blob_watched, handler):
    assert type(blob_watched) is remote_blob_util.ConsistentBlob
    session_ctx.GetDefaultSession().uuid2watch_handler[uuid] = (blob_watched, handler)

class _Watcher(oneflow_internal.ForeignWatcher):
    def __init__(self):
        oneflow_internal.ForeignWatcher.__init__(self)

    def Call(self, handler_uuid, of_blob_ptr):
        try:
            _WatcherHandler(handler_uuid, of_blob_ptr)
        except Exception as e:
            print(traceback.format_exc())
            raise e

def _WatcherHandler(handler_uuid, of_blob_ptr):
    uuid2handler = session_ctx.GetDefaultSession().uuid2watch_handler
    assert handler_uuid in uuid2handler
    blob_watched, handler = uuid2handler[handler_uuid]
    assert callable(handler)
    ndarray_lists = ofblob.OfBlob(of_blob_ptr).CopyToNdarrayLists()
    handler(local_blob_util.MakeLocalBlob(ndarray_lists, blob_watched))

_global_watcher = _Watcher()
c_api_util.RegisterWatcherOnlyOnce(_global_watcher)
