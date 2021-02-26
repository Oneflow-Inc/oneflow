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
It's copied from 
https://github.com/numba/llvmlite/blob/3701287/llvmlite/binding/common.py
"""
import atexit


# module globals may be cleared at shutdown,
# so declare this variable as an array to avoid "global _shutting_down"
_shutting_down = [False]


def SetShuttingDown():
    _shutting_down[0] = True


def IsShuttingDown(_shutting_down=_shutting_down):
    """
    Whether the interpreter is currently shutting down.
    For use in finalizers, __del__ methods, and similar; it is advised
    to early bind this function rather than look it up when calling it,
    since at shutdown module globals may be cleared.
    """
    return _shutting_down[0]
