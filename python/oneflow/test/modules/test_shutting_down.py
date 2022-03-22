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
import oneflow


class TestCallWhenShuttingDown:
    def __init__(self):
        tensor = oneflow.ones((2, 2))
        print(tensor)

    def __del__(self):
        tensor = oneflow.ones((2, 2))
        print(tensor)


test_call_when_shutting_down = TestCallWhenShuttingDown()


class TestSyncWhenShuttingDown:
    def __del__(self):
        oneflow._oneflow_internal.eager.Sync()


test_sync_when_shutting_down = TestSyncWhenShuttingDown()
