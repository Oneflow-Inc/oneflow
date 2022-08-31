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
import oneflow._oneflow_internal


class thread:
    r"""Context-manager to pick worker thread.
    By default, all opkernels are excuted/launched in worker thread 0. Within this context, opkernels can be excuted/launched in the worker thread indicated by `thread_global_id`. 
    This context manager is thread local; it will not affect ops in other threads.
    Also functions as a decorator. (Make sure to instantiate with parenthesis.)

    Args:
        thread_global_id: a integer in range [0, 3) for communication across ranks.

    For example:

    .. code-block:: python
        >>> import oneflow as flow
        >>> with flow.asyncs.thread(2):
        ...     print(flow.ones(2, 2))
        ...
        tensor([[1., 1.],
                [1., 1.]], dtype=oneflow.float32)
        >>> @flow.asyncs.thread(3)
        ... def ones_in_other_thread():
        ...     return flow.ones(2, 2)
        ...
        >>> print(ones_in_other_thread())
        tensor([[1., 1.],
                [1., 1.]], dtype=oneflow.float32)

    """

    def __init__(self, thread_global_id: int = 1, exclude_ccl=False):
        self.stream_set_ = oneflow._oneflow_internal.StreamSet(
            thread_global_id, stream_set_id=0
        )
        self.exclude_ccl_ = exclude_ccl

    def __enter__(self):
        self.guard_ = oneflow._oneflow_internal.StreamGuard(
            self.stream_set_, self.exclude_ccl_
        )

    def __exit__(self, type, value, traceback):
        del self.guard_

    def __call__(self, func):
        def Func(*args, **kwargs):
            guard = oneflow._oneflow_internal.StreamGuard(
                self.stream_set_, self.exclude_ccl_
            )
            return func(*args, **kwargs)
            del guard

        return Func
