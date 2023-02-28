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
from oneflow.profiler.profiler import (
    profile,
    record_function,
    ProfilerActivity,
    ProfilerAction,
    tensorboard_trace_handler,
)

__all__ = [
    "range_push",
    "range_pop",
    "profiler_start",
    "profiler_stop",
    "profile",
    "record_function",
    "ProfilerActivity",
    "kineto_available",
    "tensorboard_trace_handler",
    "ProfilerAction",
]


def range_push(range_name):
    oneflow._oneflow_internal.profiler.RangePush(range_name)


def range_pop():
    oneflow._oneflow_internal.profiler.RangePop()


def profiler_start():
    oneflow._oneflow_internal.profiler.ProfilerStart()


def profiler_stop():
    oneflow._oneflow_internal.profiler.ProfilerStop()


def kineto_available():
    return True
