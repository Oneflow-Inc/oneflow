from __future__ import absolute_import

from contextlib import contextmanager
import oneflow.python.framework.watch_scope_context as watch_scope_context
import oneflow.python.lib.core.traceback as traceback
from oneflow.python.oneflow_export import oneflow_export

import oneflow

class WatchScope(object):
    def __init__(self, blob_watcher_dict, diff_blob_watcher_dict = None):
        assert isinstance(blob_watcher_dict, dict)
        if diff_blob_watcher_dict is not None: assert isinstance(diff_blob_watcher_dict, dict)
        self.blob_watcher_dict_ = blob_watcher_dict
        self.diff_blob_watcher_dict_ = diff_blob_watcher_dict

    def Watch(self, blob_def):
        lbn = blob_def.logical_blob_name
        oneflow.watch(blob_def, _MakeStoreBlobCallback(self.blob_watcher_dict_, lbn))
        if self.diff_blob_watcher_dict_ is not None:
            oneflow.watch_diff(blob_def, _MakeStoreBlobCallback(self.diff_blob_watcher_dict_, lbn))

#@oneflow_export("watch_scope")
@contextmanager
def watch_scope(blob_watcher_dict, diff_blob_watcher_dict = None):
    watch_scope_context.WatcherScopeStackPush(WatchScope(blob_watcher_dict,
                                                           diff_blob_watcher_dict))
    yield
    watch_scope_context.WatcherScopeStackPop()

def _MakeStoreBlobCallback(storage, lbn):
    stack_info = traceback.GetStackInfo()[0:-1]
    def StoreFunc(x):
        storage[lbn] = {}
        storage[lbn]["location"] = stack_info
        storage[lbn]["blob"] = x
    return StoreFunc
