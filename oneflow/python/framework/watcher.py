"""
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from __future__ import absolute_import

import traceback

import oneflow.core.record.record_pb2 as record_util
import oneflow.python.framework.local_blob as local_blob_util
import oneflow.python.framework.ofblob as ofblob
import oneflow.python.framework.remote_blob as remote_blob_util
import oneflow.python.framework.session_context as session_ctx
import oneflow.python.framework.typing_util as oft_util
import oneflow_api
from google.protobuf import text_format


def BindUuidAndHandler(uuid, blob_watched, handler):
    assert isinstance(blob_watched, oneflow_api.ConsistentBlob)
    session_ctx.GetDefaultSession().uuid2watch_handler[uuid] = (blob_watched, handler)


class _Watcher(oneflow_api.ForeignWatcher):
    def __init__(self):
        oneflow_api.ForeignWatcher.__init__(self)

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
    ndarray = ofblob.OfBlob(of_blob_ptr).CopyToNdarray()
    local_blob = local_blob_util.MakeLocalBlob(ndarray, blob_watched)
    handler(oft_util.TransformWatchedBlob(local_blob, handler))


# static lifetime
_global_watcher = _Watcher()
oneflow_api.RegisterWatcherOnlyOnce(_global_watcher)
