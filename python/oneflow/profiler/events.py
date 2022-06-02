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
from typing import Tuple, Dict, Optional
from collections import OrderedDict
from prettytable import PrettyTable
from oneflow.profiler.util import format_time


def format_event_type(event_type, on_gpu: bool):
    if event_type == 0:
        return "custom"
    if event_type == 1:
        return "kernel" + ("@gpu" if on_gpu else "@cpu")
    raise ValueError(f"Undefined event type {event_type}.")


class Event:
    def __init__(
        self,
        name: str,
        cpu_time: float,
        gpu_time: Optional[float],
        bandwidth: Optional[int],
        count: int,
        input_shapes: str,
        event_type: int,
    ) -> None:
        self.name = name
        self.cpu_time = cpu_time
        self.cpu_time_total = cpu_time * count

        self.gpu_time = gpu_time
        self.gpu_time_total = gpu_time * count if self.on_gpu else None
        self.bandwidth = bandwidth
        self.bandwidth_total = bandwidth * count if self.bandwidth_is_recorded else None

        self.count = count
        self.input_shapes = input_shapes
        self.event_type = event_type
        if self.event_type == 0:
            assert not self.on_gpu, "custom events are only supported on CPU."

    @property
    def on_gpu(self) -> bool:
        return self.gpu_time is not None

    @property
    def bandwidth_is_recorded(self) -> bool:
        return self.on_gpu and self.bandwidth is not None

    def update(self, event):
        assert self.event_type == event.event_type
        assert self.on_gpu == event.on_gpu

        self.count += 1
        self.cpu_time_total += event.cpu_time
        self.cpu_time = self.cpu_time_total / self.count
        if self.on_gpu:
            self.gpu_time_total += event.gpu_time
            self.gpu_time = self.gpu_time_total / self.count
            if self.bandwidth_is_recorded:
                self.bandwidth_total += event.bandwidth
                self.bandwidth = self.bandwidth_total / self.count

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented

        return (
            self.name == other.name
            and self.on_gpu == other.on_gpu
            and self.bandwidth_is_recorded == other.bandwidth_is_recorded
            and self.cpu_time == other.cpu_time
            and self.cpu_time_total == other.cpu_time_total
            and self.gpu_time == other.gpu_time
            and self.gpu_time_total == other.gpu_time_total
            and self.bandwidth == other.bandwidth
            and self.bandwidth_total == other.bandwidth_total
            and self.count == other.count
            and self.input_shapes == other.input_shapes
            and self.event_type == other.event_type
        )

    @classmethod
    def from_dict(cls, d: dict):
        return cls(
            d.get("name"),
            d.get("cpu_time"),
            d.get("gpu_time"),
            d.get("bandwidth"),
            1,
            d.get("input_shapes"),
            d.get("type"),
        )


class Events(list):
    def __init__(self, events: str = "") -> None:
        list.__init__([])
        if events != "":
            self.__init_events(events)

    def __init_events(self, events: str):
        events_json = json.loads(events)
        for event_json in events_json:
            self.append(Event.from_dict(event_json))

    def __str__(self):
        return self.table()

    def key_averages(self):
        stats: Dict[Tuple[str, ...], Event] = OrderedDict()

        def get_key(event: Event) -> Tuple[str, ...]:
            return event.name, event.input_shapes

        for event in self:
            key = get_key(event=event)
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
            "Bandwidth",
            "Number of calls",
            "Event type",
            "Shapes of inputs",
        ]
        for item in self:
            t.add_row(
                [
                    item.name,
                    format_time(item.cpu_time_total),
                    format_time(item.cpu_time),
                    format_time(item.gpu_time_total) if item.on_gpu else "-",
                    format_time(item.gpu_time) if item.on_gpu else "-",
                    f"{item.bandwidth:.3f}GB/s" if item.bandwidth_is_recorded else "-",
                    item.count,
                    format_event_type(item.event_type, item.on_gpu),
                    item.input_shapes,
                ]
            )
        return t.get_string()
