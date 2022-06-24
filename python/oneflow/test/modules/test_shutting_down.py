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
import os


world_size = os.getenv("WORLD_SIZE")


class _TestCallWhenShuttingDown:
    def __init__(self):
        self.oneflow = oneflow
        tensor = oneflow.ones((2, 2))
        print(tensor)

    def __del__(self, of=oneflow):
        try:
            if world_size == 1:
                tensor = of.ones((2, 2))
        except:
            # Please refer to: https://github.com/Oneflow-Inc/OneTeam/issues/1219#issuecomment-1092370402
            print("__del__ at shutting down phase in Python is not stable.")


test_call_when_shutting_down = _TestCallWhenShuttingDown()


class _TestSyncWhenShuttingDown:
    def __init__(self):
        self.eager = oneflow._oneflow_internal.eager

    def __del__(self):
        try:
            self.eager.Sync()
        except:
            # Please refer to: https://github.com/Oneflow-Inc/OneTeam/issues/1219#issuecomment-1092370402
            print("__del__ at shutting down phase in Python is not stable.")


test_sync_when_shutting_down = _TestSyncWhenShuttingDown()
