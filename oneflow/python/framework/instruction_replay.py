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
import oneflow._oneflow_internal
from oneflow.python.oneflow_export import oneflow_export


@oneflow_export("start_recording_instruction")
def start_recording_instruction():
    oneflow._oneflow_internal.start_recording_instruction()


@oneflow_export("end_and_clear_recording_instruction")
def end_and_clear_recording_instruction():
    oneflow._oneflow_internal.end_and_clear_recording_instruction()


@oneflow_export("replay_instruction")
def replay_instruction():
    oneflow._oneflow_internal.replay_instruction()
