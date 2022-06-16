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
import copy
from enum import Enum
from typing import Tuple, List, Dict
from collections import OrderedDict
from prettytable import PrettyTable
from oneflow.profiler.util import format_time


class EventType(Enum):
    Custom = 0
    Kernel = 1


class CustomEventType(Enum):
    Default = 0
    CudaKernel = 1
    CudaRuntime = 2


class EventBase:
    def __init__(self, name: str, time: float, event_type: EventType) -> None:
        self._name = name
        self.time = time
        self.time_total = self.time
        self.count = 1
        self.event_type = event_type

    def update(self, event):
        assert self.event_type == event.event_type
        self.count += 1
        self.time_total += event.time
        self.time = self.time_total / self.count

    @property
    def name(self):
        if len(self._name) > 55:
            return self._name[:52] + "..."
        return self._name


class CustomEvent(EventBase):
    def __init__(
        self, name: str, time: float, custom_event_type: CustomEventType
    ) -> None:
        super().__init__(name, time, EventType.Custom)
        self.custom_event_type = custom_event_type

    @classmethod
    def from_dict(cls, d: dict):
        return cls(d.get("name"), d.get("time"), CustomEventType(d.get("custom_type")))

    @property
    def key(self):
        return self.name, self.custom_event_type

    @property
    def time_is_on_cuda(self):
        return self.custom_event_type == CustomEventType.CudaKernel

    @property
    def row(self):
        if self.time_is_on_cuda:
            return (
                self.name,
                "_",
                "_",
                format_time(self.time_total),
                format_time(self.time),
                self.count,
                "-",
                "-",
            )
        return (
            self.name,
            format_time(self.time_total),
            format_time(self.time),
            "-",
            "-",
            self.count,
            "-",
            "-",
        )


class KernelEvent(EventBase):
    def __init__(
        self, name: str, time: float, memory_size: int, input_shapes: str
    ) -> None:
        super().__init__(name, time, EventType.Kernel)
        self.children: List[CustomEvent] = []
        self.memory_size = memory_size
        self.input_shapes = input_shapes
        self.cuda_time_total = 0.0
        self.cuda_time = 0.0

    @classmethod
    def from_dict(cls, d: dict):
        kernel_event = cls(
            d.get("name"), d.get("time"), d.get("memory_size"), d.get("input_shapes")
        )
        if "children" in d.keys():
            children_list = d.get("children")
            if len(children_list) > 0:
                for child_dict in children_list:
                    kernel_event.children.append(CustomEvent.from_dict(child_dict))
        return kernel_event

    @property
    def key(self):
        if not self.on_gpu:
            return (self.name, self.input_shapes)
        return (
            self.name,
            self.input_shapes,
            ",".join([x.name for x in self.children]),
        )

    @property
    def has_cuda_time(self):
        return self.cuda_time_total > 0.0

    @property
    def bandwidth(self):
        if self.on_gpu and self.has_cuda_time:
            if self.memory_size != -1:
                return f"{self.memory_size / (1024.0 * 1024.0 * 1024.0) / (self.cuda_time / (1000 * 1000)):.3f}GB/s"
        return "-"

    @property
    def row(self):
        return (
            self.name,
            format_time(self.time_total),
            format_time(self.time),
            format_time(self.cuda_time_total) if self.has_cuda_time else "-",
            format_time(self.cuda_time) if self.has_cuda_time else "-",
            self.count,
            self.input_shapes,
            self.bandwidth,
        )

    @property
    def children_rows(self):
        results = []
        for item in self.children:
            results.append(item.row)
        return results

    @property
    def on_gpu(self) -> bool:
        return len(self.children) > 0

    def update(self, event):
        assert isinstance(event, type(self))
        assert self.on_gpu == event.on_gpu
        assert self.key == event.key
        super().update(event)
        self.cuda_time_total += event.cuda_time_total
        self.cuda_time = self.cuda_time_total / self.count
        for i in range(len(self.children)):
            self.children[i].update(event.children[i])

    def make_children_average(self):
        stats: Dict[Tuple[str, ...], CustomEvent] = OrderedDict()
        for event in self.children:
            if event.key in stats:
                stats[event.key].update(event)
            else:
                stats[event.key] = copy.deepcopy(event)
        self.children = list(stats.values())
        for event in self.children:
            if event.time_is_on_cuda:
                self.cuda_time += event.time
        self.cuda_time_total = self.cuda_time
        self.children.sort(key=lambda x: x.name)


class Events(list):
    def __init__(self, events: str = "") -> None:
        list.__init__([])
        if events != "":
            self.__init_events(events)

    def __init_events(self, events: str):
        events_json = json.loads(events)
        classes = [CustomEvent, KernelEvent]
        for event_json in events_json:
            self.append(classes[event_json.get("type")].from_dict(event_json))

    def __str__(self):
        return self.table()

    def key_averages(self):
        stats: Dict[Tuple[str, ...], EventBase] = OrderedDict()

        for event in self:
            if isinstance(event, KernelEvent) and event.on_gpu:
                event.make_children_average()
            key = event.key
            if key in stats:
                stats[key].update(event)
            else:
                stats[key] = copy.deepcopy(event)
        results = Events()
        results.extend(stats.values())
        return results

    def table(self):
        t = PrettyTable()
        t.field_names = [
            "Name",
            "CPU time total",
            "CPU time",
            "GPU time total",
            "GPU time",
            "Number of calls",
            "Shapes of inputs",
            "Bandwidth",
        ]
        for item in self:
            if isinstance(item, CustomEvent):
                t.add_row(item.row)
            if isinstance(item, KernelEvent):
                t.add_row(item.row)
                if item.on_gpu:
                    t.add_rows(item.children_rows)
        return t.get_string()
