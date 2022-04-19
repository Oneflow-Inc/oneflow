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
import json
from prettytable import PrettyTable
from oneflow.framework.profiler import ProfilerStart as profiler_start
from oneflow.framework.profiler import ProfilerStop as profiler_stop
from oneflow.framework.profiler import RangePop as range_pop
from oneflow.framework.profiler import RangePush as range_push
from oneflow.framework.profiler import EnableProfiler as enable_profiler
from oneflow.framework.profiler import DisableProfiler as disable_profiler
from oneflow.framework.profiler import StartRecord as start_record
from oneflow.framework.profiler import EndRecord as end_record


class profile(object):
    def __init__(self) -> None:
        self.result = ""

    def __enter__(self):
        enable_profiler()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.result = disable_profiler()

    # copy from pytorch: torch/autograd/profiler_util.py
    def _format_time(self, time_ns):
        NS_IN_SECOND = 1000.0 * 1000.0 * 1000.0
        NS_IN_MS = 1000.0 * 1000.0
        NS_IN_US = 1000.0
        if time_ns >= NS_IN_SECOND:
            return "{:.3f}s".format(time_ns / NS_IN_SECOND)
        if time_ns >= NS_IN_MS:
            return "{:.3f}ms".format(time_ns / NS_IN_MS)
        if time_ns >= NS_IN_US:
            return "{:.3f}us".format(time_ns / NS_IN_US)
        return "{:.3f}us".format(time_ns)

    def table(self):
        result_json = json.loads(self.result)
        t = PrettyTable()
        t.field_names = ["Name", "All duration", "Average duration", "Number of calls"]
        for item in result_json:
            t.add_row(
                [
                    item["op_name"],
                    self._format_time(item["all_duration"]),
                    self._format_time(item["avg_duration"]),
                    item["num_called"],
                ]
            )
        return t.get_string()


class record_function(object):
    def __init__(self, name: str) -> None:
        self.name = name

    def __enter__(self):
        start_record(self.name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_record(self.name)

