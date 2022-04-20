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
"""
oneflow.multiprocessing is a wrapper around the native :mod:`multiprocessing`
module. It registers custom reducers, that use shared memory to provide shared
views on the same data in different processes. Once the tensor/storage is moved
to shared_memory (see :func:`~oneflow.Tensor.share_memory_`), it will be possible
to send it to other processes without making any copies.

The API is 100% compatible with the original module - it's enough to change
``import multiprocessing`` to ``import oneflow.multiprocessing`` to have all the
tensors sent through the queues or shared via other mechanisms, moved to shared
memory.

Because of the similarity of APIs we do not document most of this package
contents, and we recommend referring to very good docs of the original module.
"""
import oneflow as flow
import sys
from .reductions import init_reductions
import multiprocessing

__all__ = [
    "set_sharing_strategy",
    "get_sharing_strategy",
    "get_all_sharing_strategies",
    "unlink_all_shared_memory",
]


from multiprocessing import *  # noqa: F403


__all__ += multiprocessing.__all__  # type: ignore[attr-defined]


# This call adds a Linux specific prctl(2) wrapper function to this module.
# See https://github.com/pytorch/pytorch/pull/14391 for more information.
flow._oneflow_internal._multiprocessing_init()


"""Add helper function to spawn N processes and wait for completion of any of
them. This depends `mp.get_context` which was added in Python 3.4."""
from .spawn import (
    spawn,
    SpawnContext,
    start_processes,
    ProcessContext,
    ProcessRaisedException,
    ProcessExitedException,
)


if sys.platform == "darwin" or sys.platform == "win32":
    _sharing_strategy = "file_system"
    _all_sharing_strategies = {"file_system"}
else:
    _sharing_strategy = "file_descriptor"
    _all_sharing_strategies = {"file_descriptor", "file_system"}


def set_sharing_strategy(new_strategy):
    """Sets the strategy for sharing CPU tensors.

    Args:
        new_strategy (str): Name of the selected strategy. Should be one of
            the values returned by :func:`get_all_sharing_strategies()`.
    """
    global _sharing_strategy
    assert new_strategy in _all_sharing_strategies
    _sharing_strategy = new_strategy


def get_sharing_strategy():
    """Returns the current strategy for sharing CPU tensors."""
    return _sharing_strategy


def get_all_sharing_strategies():
    """Returns a set of sharing strategies supported on a current system."""
    return _all_sharing_strategies


def unlink_all_shared_memory():
    flow._oneflow_internal.multiprocessing.unlink_all_shared_memory()


init_reductions()
