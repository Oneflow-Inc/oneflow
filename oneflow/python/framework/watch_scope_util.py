from __future__ import absolute_import

from contextlib import contextmanager

import oneflow
import oneflow.python.framework.watch_scope_context as watch_scope_context
import oneflow.python.lib.core.traceinfo as traceinfo
from oneflow.python.oneflow_export import oneflow_export


def TryWatchOnce(blob_def):
    for watch_scope in watch_scope_context.EachWatchScope():
        watch_scope.WatchOnce(blob_def)


class WatchScope(object):
    def __init__(self, blob_watcher=None, diff_blob_watcher=None):
        if blob_watcher is not None:
            assert isinstance(blob_watcher, dict) or callable(blob_watcher)
        self.watched_blob_lbn = set()
        self.blob_watcher_ = blob_watcher
        if diff_blob_watcher is not None:
            assert isinstance(diff_blob_watcher, dict) or callable(diff_blob_watcher)
        self.diff_blob_watcher_ = diff_blob_watcher

    def WatchOnce(self, blob_def):
        if blob_def.unique_name in self.watched_blob_lbn:
            return
        if self.blob_watcher_ is not None:
            oneflow.watch(
                blob_def, _MakeBlobWatchCallback(self.blob_watcher_, blob_def)
            )
        if self.diff_blob_watcher_ is not None:
            oneflow.watch_diff(
                blob_def, _MakeBlobWatchCallback(self.diff_blob_watcher_, blob_def)
            )
        self.watched_blob_lbn.add(blob_def.unique_name)


@oneflow_export("watch_scope")
@contextmanager
def watch_scope(blob_watcher=None, diff_blob_watcher=None):
    watch_scope_context.WatchScopeStackPush(WatchScope(blob_watcher, diff_blob_watcher))
    yield
    watch_scope_context.WatchScopeStackPop()


def _MakeBlobWatchCallback(storage_or_func, blob_def):
    if isinstance(storage_or_func, dict):
        storage = storage_or_func

        def StoreFunc(blob):
            storage[blob_def.unique_name] = dict(blob=blob, blob_def=blob_def)

        return StoreFunc
    elif callable(storage_or_func):
        func = storage_or_func
        return lambda blob: func(blob, blob_def)
    else:
        raise NotImplementedError
