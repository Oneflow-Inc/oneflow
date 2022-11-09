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
import sys

__all__ = ["register_after_fork"]

if sys.platform == "win32" or sys.version_info < (3, 7):
    import multiprocessing.util as _util

    def _register(func):
        def wrapper(arg):
            func()

        _util.register_after_fork(_register, wrapper)


else:
    import os

    def _register(func):
        os.register_at_fork(after_in_child=func)


def register_after_fork(func):
    """Register a callable to be executed in the child process after a fork.

    Note:
        In python < 3.7 this will only work with processes created using the
        ``multiprocessing`` module. In python >= 3.7 it also works with
        ``os.fork()``.

    Args:
        func (function): Function taking no arguments to be called in the child after fork

    """
    _register(func)
