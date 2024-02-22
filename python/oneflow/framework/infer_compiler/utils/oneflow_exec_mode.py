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
import oneflow as flow

_ONEFLOW_EXEC_MODE = False


class oneflow_exec_mode(object):
    def __init__(self, enabled=None):
        if enabled is not None:
            self.enabled = enabled
        else:
            self.enabled = True

    def __enter__(self):
        global _ONEFLOW_EXEC_MODE
        self.prev_mode = _ONEFLOW_EXEC_MODE
        _ONEFLOW_EXEC_MODE = self.enabled
        self.prev_grad_mode = flow.is_grad_enabled()
        _ = flow.set_grad_enabled(False)

    def __exit__(self, exc_type, exc_val, exc_tb):
        global _ONEFLOW_EXEC_MODE
        _ONEFLOW_EXEC_MODE = self.prev_mode
        _ = flow.set_grad_enabled(self.prev_grad_mode)


def oneflow_exec_mode_enabled():
    global _ONEFLOW_EXEC_MODE
    return _ONEFLOW_EXEC_MODE
