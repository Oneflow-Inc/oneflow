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
from enum import Enum
from typing import Optional, Iterable, Set
from oneflow.profiler.events import Events


class ProfilerActivity(Enum):
    CPU = 1
    CUDA = 2


class ProfilerAction(Enum):
    """
    Profiler actions that can be taken at the specified intervals
    """

    NONE = 0
    WARMUP = 1
    RECORD = 2
    RECORD_AND_SAVE = 3


def tensorboard_trace_handler():
    raise NotImplementedError()


def supported_activities() -> Set[ProfilerActivity]:
    activities = set([ProfilerActivity.CPU])
    if oneflow.cuda.is_available():
        activities.add(ProfilerActivity.CUDA)
    return activities


class profile:
    def __init__(
        self,
        activities: Optional[Iterable[ProfilerActivity]] = None,
        record_shapes: bool = False,
        record_attrs: bool = False,
        record_bandwidth_for_cuda: bool = False,
    ) -> None:
        self.activities = set(activities) if activities else supported_activities()
        assert (
            len(self.activities) > 0
        ), "At least one ProfilerActivity must be specified."
        for item in self.activities:
            assert (
                item in supported_activities()
            ), f"Unsupported ProfilerActivity {item}"
        self.record_shapes = record_shapes
        self.record_attrs = record_attrs
        if not (ProfilerActivity.CUDA in self.activities):
            assert (
                record_bandwidth_for_cuda == False
            ), "record_bandwidth_for_cuda = True can only work with cuda."
        self.record_bandwidth_for_cuda = record_bandwidth_for_cuda
        self.profile_events: Optional[Events] = None

    def __enter__(self):
        oneflow._oneflow_internal.profiler.EnableProfiler(
            ProfilerActivity.CPU in self.activities,
            ProfilerActivity.CUDA in self.activities,
            self.record_shapes,
            self.record_attrs,
            self.record_bandwidth_for_cuda,
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.profile_events = Events(
            oneflow._oneflow_internal.profiler.DisableProfilerAndReturnResult()
        )

    def __check_finish(self):
        if self.profile_events is None:
            raise RuntimeError("Profiler didn't finish running")

    def key_averages(self, group_by_input_shape=False, group_by_attributes=False):
        self.__check_finish()
        return self.profile_events.key_averages(
            group_by_input_shape=group_by_input_shape,
            group_by_attributes=group_by_attributes,
        )

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
