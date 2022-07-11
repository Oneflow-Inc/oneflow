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
from rich import box
from rich.console import Console
from rich.table import Table
from oneflow.profiler.util import format_time


class EventType(Enum):
    Custom = 0
    Kernel = 1


class CustomEventType(Enum):
    Default = 0
    CudaKernel = 1
    CudaRuntime = 2


class EventBase:
    MAX_NAME_LENGTH = 55

    def __init__(self, name: str, time_total: float, event_type: EventType) -> None:
        self._name: str = name
        self._time_total: float = time_total
        self.count: int = 1
        self.event_type: EventType = event_type

    def update(self, event) -> None:
        assert self.event_type == event.event_type
        self.cpu_time_total += event.cpu_time_total
        self.count += event.count

    @property
    def name(self):
        if len(self._name) > self.MAX_NAME_LENGTH:
            return self._name[: self.MAX_NAME_LENGTH - 3] + "..."
        return self._name

    @property
    def cpu_time_total(self):
        return self._time_total

    @cpu_time_total.setter
    def cpu_time_total(self, new_time):
        self._time_total = new_time

    @property
    def cpu_time(self):
        return self._time_total / self.count

    @property
    def cuda_time_total(self):
        return None

    @cuda_time_total.setter
    def cuda_time_total(self, new_time):
        pass

    @property
    def cuda_time(self):
        if self.cuda_time_total is None:
            return None
        return self.cuda_time_total / self.count

    def has_cuda_time(self) -> bool:
        return self.cuda_time_total is not None

    def __eq__(self, __o: object) -> bool:
        return (
            self.name == __o.name
            and self.count == __o.count
            and self.cpu_time_total == __o.cpu_time_total
            and self.cuda_time_total == __o.cuda_time_total
        )


class CustomEvent(EventBase):
    def __init__(
        self, name: str, time_total: float, custom_event_type: CustomEventType
    ) -> None:
        super().__init__(name, time_total, EventType.Custom)
        self.custom_event_type = custom_event_type

    @classmethod
    def from_dict(cls, d: dict):
        return cls(d.get("name"), d.get("time"), CustomEventType(d.get("custom_type")))

    @property
    def key(self):
        return self.name, self.custom_event_type

    @property
    def cuda_time_total(self):
        if self.custom_event_type == CustomEventType.CudaKernel:
            return self._time_total
        return None

    def to_dict(self):
        device_prefix = "cuda" if self.has_cuda_time() else "cpu"
        time_attrs = [f"{device_prefix}_{suffix}" for suffix in ["time", "time_total"]]
        result = {
            "name": self.name,
            "count": self.count,
        }
        for time_attr in time_attrs:
            result[time_attr] = format_time(getattr(self, time_attr))
        return result

    def __eq__(self, __o: object) -> bool:
        return (
            super().__eq__(__o)
            and isinstance(__o, type(self))
            and self.custom_event_type == __o.custom_event_type
        )


class KernelEvent(EventBase):
    def __init__(
        self,
        name: str,
        time_total: float,
        memory_size: int,
        description: Dict[str, str],
    ) -> None:
        super().__init__(name, time_total, EventType.Kernel)
        self.children: List[CustomEvent] = []
        self.memory_size = memory_size
        self.description = description
        self._cuda_time_total = 0.0
        self._enable_show_input_shapes = True
        self._enable_show_attributes = True

    def add_child(self, event: CustomEvent):
        self.children.append(event)
        if event.has_cuda_time():
            self._cuda_time_total += event.cuda_time

    @classmethod
    def from_dict(cls, d: dict):
        kernel_event = cls(
            d.get("name"), d.get("time"), d.get("memory_size"), d.get("description", {})
        )
        if "children" in d.keys():
            children_list = d.get("children")
            if len(children_list) > 0:
                for child_dict in children_list:
                    kernel_event.add_child(CustomEvent.from_dict(child_dict))
        return kernel_event

    @property
    def key(self):
        def get_extra_keys():
            extra_keys = []
            if self.input_shapes != "" and self._enable_show_input_shapes:
                extra_keys.append(self.description.get("input_shapes")[1])
            if self.attributes != "" and self._enable_show_attributes:
                extra_keys.append(self.description.get("attrs")[1])
            return tuple(extra_keys)

        if len(self.children) == 0:
            return (self.name,) + get_extra_keys()
        return (
            self.name,
            *get_extra_keys(),
            ",".join([x.name for x in self.children]),
        )

    @property
    def cuda_time_total(self):
        if self._cuda_time_total > 0.0:
            return self._cuda_time_total
        return None

    @cuda_time_total.setter
    def cuda_time_total(self, new_time):
        self._cuda_time_total = new_time

    @property
    def input_shapes(self):
        if "input_shapes" in self.description:
            return self.description["input_shapes"][0]
        return ""

    @property
    def attributes(self):
        if "attrs" in self.description:
            return self.description["attrs"][0]
        return ""

    @property
    def bandwidth(self):
        if len(self.children) > 0 and self.has_cuda_time():
            if self.memory_size != -1:
                return f"{self.memory_size / (1024.0 * 1024.0 * 1024.0) / (self.cuda_time / (1000 * 1000)):.3f}GB/s"
        return ""

    def to_dict(self):
        result = {
            "name": self.name,
            "cpu_time_total": format_time(self.cpu_time_total),
            "cpu_time": format_time(self.cpu_time),
            "count": self.count,
            "input_shapes": self.input_shapes,
            "attributes": self.attributes,
        }
        if self.has_cuda_time():
            result.update(
                {
                    "cuda_time_total": format_time(self.cuda_time_total),
                    "cuda_time": format_time(self.cuda_time),
                }
            )

        return result

    def update(self, event):
        assert id(self) != id(event)
        assert isinstance(event, type(self))
        assert len(self.children) == len(event.children)
        assert self.has_cuda_time() == event.has_cuda_time()
        assert self.key == event.key

        super().update(event)
        if self.has_cuda_time():
            self.cuda_time_total += event.cuda_time_total

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
        self.children.sort(key=lambda x: x.name)

    def __eq__(self, __o: object) -> bool:
        return (
            super().__eq__(__o)
            and isinstance(__o, type(self))
            and self.children == __o.children
            and self.memory_size == __o.memory_size
            and self.input_shapes == __o.input_shapes
            and self.attributes == __o.attributes
        )


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

    def key_averages(self, group_by_input_shape=False, group_by_attributes=False):
        stats: Dict[Tuple[str, ...], EventBase] = OrderedDict()

        def deal_event(e):
            if isinstance(e, KernelEvent):
                e._enable_show_input_shapes = group_by_input_shape
                e._enable_show_attributes = group_by_attributes

            key = e.key
            if key in stats:
                stats[key].update(e)
            else:
                stats[key] = copy.deepcopy(e)

        for event in self:
            if isinstance(event, KernelEvent) and len(event.children) != 0:
                event.make_children_average()
                for event_child in event.children:
                    deal_event(event_child)
                event.children = []
            deal_event(event)

        results = Events()
        results.extend(stats.values())
        return results

    def table(self):
        has_input_shapes = any(
            [
                x.input_shapes != "" and x._enable_show_input_shapes
                for x in self
                if isinstance(x, KernelEvent)
            ]
        )
        has_attributes = any(
            [
                x.attributes != "" and x._enable_show_attributes
                for x in self
                if isinstance(x, KernelEvent)
            ]
        )
        has_bandwidth = any(
            [x.bandwidth != "" for x in self if isinstance(x, KernelEvent)]
        )
        t = Table(
            "Name",
            "CPU time total",
            "CPU time",
            "GPU time total",
            "GPU time",
            "Number of calls",
            box=box.SIMPLE,
        )
        field_keys = [
            "name",
            "cpu_time_total",
            "cpu_time",
            "cuda_time_total",
            "cuda_time",
            "count",
        ]
        if has_input_shapes:
            t.add_column("Input shapes")
            field_keys.append("input_shapes")
        if has_attributes:
            t.add_column("Attributes")
            field_keys.append("attributes")
        if has_bandwidth:
            t.add_column("Bandwidth")
            field_keys.append("bandwidth")

        def build_row(data: dict):
            return tuple(str(data.get(key, "")) for key in field_keys)

        for item in self:
            if isinstance(item, CustomEvent):
                t.add_row(*build_row(item.to_dict()))
            if isinstance(item, KernelEvent):
                t.add_row(*build_row(item.to_dict()))
                if len(item.children) > 0:
                    for child in item.children:
                        t.add_row(*build_row(child.to_dict()))
        console = Console()
        with console.capture() as capture:
            console.print(t)
        return capture.get()
