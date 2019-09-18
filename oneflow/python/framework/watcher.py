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

    def Call(self, handler_uuid, int64_list_serialized_ofblob_ptrs):
        try:
            _WatcherHandler(handler_uuid, int64_list_serialized_ofblob_ptrs)
        except Exception as e:
            print(traceback.format_exc())
            raise e

def _WatcherHandler(handler_uuid, int64_list_serialized_ofblob_ptrs):
    int_list = text_format.Parse(int64_list_serialized_ofblob_ptrs, record_util.Int64List())
    tensors = [ofblob.OfBlob(of_blob_ptr).CopyToNdarray() for of_blob_ptr in int_list.value]
    assert handler_uuid in _uuid2handler
    _uuid2handler[handler_uuid](*tensors)

_global_watcher = _Watcher()
c_api_util.RegisterWatcherOnlyOnce(_global_watcher)

_uuid2handler = dict()
