from __future__ import absolute_import

from google.protobuf import text_format
import oneflow.core.record.record_pb2 as record_util
import oneflow.python.framework.ofblob as ofblob
import oneflow.oneflow_internal as oneflow_internal
import oneflow.python.framework.c_api_util as c_api_util
import traceback

def BindUuidAndHandler(uuid, handler):
     assert uuid not in _uuid2handler
     _uuid2handler[uuid] = handler

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
    assert handler_uuid in _uuid2handler
    _uuid2handler[handler_uuid](ofblob.OfBlob(of_blob_ptr).CopyToBlob())

_global_watcher = _Watcher()
c_api_util.RegisterWatcherOnlyOnce(_global_watcher)

_uuid2handler = dict()
