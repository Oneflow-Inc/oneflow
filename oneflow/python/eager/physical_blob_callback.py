from __future__ import absolute_import

from google.protobuf import text_format
import oneflow.core.record.record_pb2 as record_util
import oneflow.python.framework.ofblob as ofblob
import oneflow.oneflow_internal as oneflow_internal
import oneflow.python.framework.c_api_util as c_api_util
import traceback

def GetIdForRegisteredCallback(cb):
    assert callable(cb)
    global unique_id2handler
    unique_id2handler[id(cb)] = cb
    return id(cb)

def DeleteRegisteredCallback(cb):
    global unique_id2handler
    assert id(cb) in unique_id2handler
    del unique_id2handler[id(cb)]

class _WorkerCallback(oneflow_internal.ForeignWorkerCallback):
    def __init__(self):
        oneflow_internal.ForeignWorkerCallback.__init__(self)

    def Call(self, unique_id, of_blob_ptr):
        try:
            _WatcherHandler(unique_id, of_blob_ptr)
        except Exception as e:
            print(traceback.format_exc())
            raise e

def _WatcherHandler(unique_id, of_blob_ptr):
    global unique_id2handler
    assert unique_id in unique_id2handler
    handler = unique_id2handler[unique_id]
    assert callable(handler)
    handler(ofblob.OfBlob(of_blob_ptr))

unique_id2handler = {}
# static lifetime 
_worker_callback = _WorkerCallback()
c_api_util.RegisterWorkerCallbackOnlyOnce(_worker_callback)
