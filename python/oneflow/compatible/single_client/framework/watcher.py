import traceback
from oneflow.compatible.single_client.core.record import record_pb2 as record_util
from oneflow.compatible.single_client.python.framework import local_blob as local_blob_util
from oneflow.compatible.single_client.python.framework import ofblob as ofblob
from oneflow.compatible.single_client.python.framework import remote_blob as remote_blob_util
from oneflow.compatible.single_client.python.framework import session_context as session_ctx
from oneflow.compatible.single_client.python.framework import typing_util as oft_util
import oneflow._oneflow_internal
from google.protobuf import text_format

def BindUuidAndHandler(uuid, blob_watched, handler):
    assert isinstance(blob_watched, oneflow._oneflow_internal.ConsistentBlob)
    session_ctx.GetDefaultSession().uuid2watch_handler[uuid] = (blob_watched, handler)

class _Watcher(oneflow._oneflow_internal.ForeignWatcher):

    def __init__(self):
        oneflow._oneflow_internal.ForeignWatcher.__init__(self)

    def Call(self, handler_uuid, of_blob_ptr):
        try:
            _WatcherHandler(handler_uuid, of_blob_ptr)
        except Exception as e:
            print(traceback.format_exc())
            raise e

def _WatcherHandler(handler_uuid, of_blob_ptr):
    uuid2handler = session_ctx.GetDefaultSession().uuid2watch_handler
    assert handler_uuid in uuid2handler
    (blob_watched, handler) = uuid2handler[handler_uuid]
    assert callable(handler)
    ndarray = ofblob.OfBlob(of_blob_ptr).CopyToNdarray()
    local_blob = local_blob_util.LocalBlob(ndarray, blob_watched.is_dynamic)
    handler(oft_util.TransformWatchedBlob(local_blob, handler))
_global_watcher = _Watcher()