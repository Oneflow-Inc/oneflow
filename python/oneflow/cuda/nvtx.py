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
from contextlib import contextmanager

import oneflow as flow


try:
    from flow._oneflow_internal import profiler
except ImportError:
    class _NVTXStub(object):
        @staticmethod
        def _fail(*args, **kwargs):
            raise RuntimeError("NVTX functions not installed. Are you sure you have a CUDA build?")

        RangePush = _fail
        RangePop = _fail
        Mark = _fail

    _nvtx = _NVTXStub()  # type: ignore[assignment]

__all__ = ['range_push', 'range_pop', 'mark', 'range']


def range_push(msg):
    """
    Pushes a range onto a stack of nested range span.  Returns zero-based
    depth of the range that is started.

    Args:
        msg (string): ASCII message to associate with range
    """
    return flow._oneflow_internal.profiler.RangePush(msg)


def range_pop():
    """
    Pops a range off of a stack of nested range spans.  Returns the
    zero-based depth of the range that is ended.
    """
    return flow._oneflow_internal.profiler.RangePop(msg)


def mark(msg):
    """
    Describe an instantaneous event that occurred at some point.

    Args:
        msg (string): ASCII message to associate with the event.
    """
    return flow._oneflow_internal.profiler.Mark(msg)


@contextmanager
def range(msg, *args, **kwargs):
    """
    Context manager / decorator that pushes an NVTX range at the beginning
    of its scope, and pops it at the end. If extra arguments are given,
    they are passed as arguments to msg.format().

    Args:
        msg (string): message to associate with the range
    """
    range_push(msg.format(*args, **kwargs))
    yield
    range_pop()


def profiler_start():
    flow._oneflow_internal.profiler.ProfilerStart()


def profiler_stop():
    flow._oneflow_internal.profiler.ProfilerStop()
