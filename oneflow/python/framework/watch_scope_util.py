from __future__ import absolute_import

from contextlib import contextmanager
import oneflow.python.framework.watch_scope_context as watch_scope_context
import oneflow.python.lib.core.traceinfo as traceinfo
from oneflow.python.oneflow_export import oneflow_export

import oneflow

def TryWatchOnce(blob_def):
    for watch_scope in watch_scope_context.EachWatchScope(): watch_scope.WatchOnce(blob_def)

class WatchScope(object):
    def __init__(self, blob_watcher_dict, diff_blob_watcher_dict = None):
        assert isinstance(blob_watcher_dict, dict)
        if diff_blob_watcher_dict is not None: assert isinstance(diff_blob_watcher_dict, dict)
        self.blob_watcher_dict_ = blob_watcher_dict
        self.diff_blob_watcher_dict_ = diff_blob_watcher_dict

    def WatchOnce(self, blob_def):
        lbn = blob_def.logical_blob_name
        if lbn in self.blob_watcher_dict_: return
        self.blob_watcher_dict_[lbn] = {}
        oneflow.watch(blob_def, _MakeStoreBlobCallback(self.blob_watcher_dict_, blob_def))
        if self.diff_blob_watcher_dict_ is not None:
            cb = _MakeStoreBlobCallback(self.diff_blob_watcher_dict_, blob_def)
            oneflow.watch_diff(blob_def, cb)

@oneflow_export("watch_scope")
@contextmanager
def watch_scope(blob_watcher_dict, diff_blob_watcher_dict = None):
    watch_scope_context.WatchScopeStackPush(WatchScope(blob_watcher_dict,
                                                         diff_blob_watcher_dict))
    yield
    watch_scope_context.WatchScopeStackPop()

def _MakeStoreBlobCallback(storage, blob_def):
    lbn = blob_def.logical_blob_name
    def StoreFunc(x):
        if lbn not in storage:
            storage[lbn] = {}
        storage[lbn]["blob_def"] = blob_def
        storage[lbn]["blob"] = x
    return StoreFunc
