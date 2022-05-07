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
from typing import Optional
from oneflow.profiler.events import Events


class profile:
    def __init__(self) -> None:
        self.profile_events: Optional[Events] = None

    def __enter__(self):
        oneflow._oneflow_internal.profiler.EnableProfiler()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.profile_events = Events(
            oneflow._oneflow_internal.profiler.DisableProfilerAndReturnResult()
        )

    def __check_finish(self):
        if self.profile_events is None:
            raise RuntimeError("Profiler didn't finish running")

    def key_averages(self):
        self.__check_finish()
        return self.profile_events.key_averages()

    def events(self):
        self.__check_finish()
        return self.profile_events


class record_function:
    def __init__(self, name: str) -> None:
        self.name = name
        self.__event_recorder_key = ""

    def __enter__(self):
        self.__event_recorder_key = oneflow._oneflow_internal.profiler.StartRecord(
            self.name
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        oneflow._oneflow_internal.profiler.EndRecord(self.__event_recorder_key)
