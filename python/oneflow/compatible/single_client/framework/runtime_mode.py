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

NORMAL_MODE = "NORMAL_MODE"
GLOBAL_MODE = "GLOBAL_MODE"
DEVICE_MODE = "DEVICE_MODE"


def CurrentMode():
    return mode_statck[0]


def IsValidMode(mode):
    return mode == NORMAL_MODE or mode == GLOBAL_MODE or mode == DEVICE_MODE


@contextmanager
def ModeScope(mode):
    global mode_statck
    mode_statck.insert(0, mode)
    try:
        yield
    finally:
        mode_statck.pop(0)


mode_statck = [NORMAL_MODE]
