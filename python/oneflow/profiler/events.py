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


def format_event_type(event_type):
    if event_type == 0:
        return "custom"
    if event_type == 1:
        return "kernel"
    raise ValueError(f"Undefined event type {event_type}.")


class Event:
    def __init__(
        self,
        name: str,
        cpu_time: int,
        cpu_time_total: int,
        count: int,
        input_shapes: str,
        event_type: int,
    ) -> None:
        self.name = name
        self.cpu_time = cpu_time
        self.cpu_time_total = cpu_time_total
        self.count = count
        self.input_shapes = input_shapes
        self.event_type = event_type

    def update(self, event):
        self.cpu_time_total += event.cpu_time
        self.count += 1
        self.cpu_time = self.cpu_time_total / self.count

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return (
            self.name == other.name
            and self.cpu_time == other.cpu_time
            and self.cpu_time_total == other.cpu_time_total
            and self.count == other.count
            and self.input_shapes == other.input_shapes
            and self.event_type == other.event_type
        )

    @classmethod
    def from_dict(cls, d: dict):
        return cls(
            d["name"], d["cpu_time"], d["cpu_time"], 1, d["input_shapes"], d["type"]
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
            "Cpu time total",
            "Cpu time",
            "Number of calls",
            "Event type",
            "Shapes of inputs",
        ]
        for item in self:
            t.add_row(
                [
                    item.name,
                    format_time(item.cpu_time_total),
                    format_time(item.cpu_time_total),
                    item.count,
                    format_event_type(item.event_type),
                    item.input_shapes,
                ]
            )
        return t.get_string()
