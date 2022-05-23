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
from typing import Tuple, Dict
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
        time: float,
        on_gpu: bool,
        bandwidth: int,
        count: int,
        input_shapes: str,
        event_type: int,
    ) -> None:
        self.name = name
        self.time = time
        self.time_total = time * count
        self.bandwidth = bandwidth
        self.bandwidth_total = bandwidth * count
        self.on_gpu = on_gpu
        self.count = count
        self.input_shapes = input_shapes
        self.event_type = event_type
        if self.event_type == 0:
            assert not self.on_gpu, "custom events are only supported on CPU."

    def update(self, event):
        assert self.event_type == event.event_type
        assert self.on_gpu == event.on_gpu

        self.time_total += event.time
        self.bandwidth_total += event.bandwidth
        self.count += 1
        self.time = self.time_total / self.count
        self.bandwidth = self.bandwidth_total / self.count

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented

        return (
            self.name == other.name
            and self.time == other.time
            and self.time_total == other.time_total
            and self.bandwidth == other.bandwidth
            and self.bandwidth_total == other.bandwidth_total
            and self.on_gpu == other.on_gpu
            and self.count == other.count
            and self.input_shapes == other.input_shapes
            and self.event_type == other.event_type
        )

    @classmethod
    def from_dict(cls, d: dict):
        return cls(
            d["name"],
            d["time"],
            d["on_gpu"],
            d["bandwidth"] if "bandwidth" in d else -1,
            1,
            d["input_shapes"],
            d["type"],
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
                    format_time(item.time_total) if not item.on_gpu else "-",
                    format_time(item.time) if not item.on_gpu else "-",
                    format_time(item.time_total) if item.on_gpu else "-",
                    format_time(item.time) if item.on_gpu else "-",
                    f"{item.bandwidth:.3f}GB/s" if item.on_gpu else "-",
                    item.count,
                    format_event_type(item.event_type, item.on_gpu),
                    item.input_shapes,
                ]
            )
        return t.get_string()
