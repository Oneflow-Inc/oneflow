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
from dataclasses import dataclass
from typing import Tuple, Dict
from collections import OrderedDict
from dacite import from_dict
from prettytable import PrettyTable
from oneflow.profiler.util import format_time


def format_event_type(event_type):
    if event_type == 0:
        return "custom"
    if event_type == 1:
        return "kernel"
    raise ValueError(f"Undefined event type {event_type}.")


@dataclass
class Event:
    name: str
    all_duration: int
    avg_duration: int
    num_called: int
    event_type: int
    shapes: str

    def __post_init__(self):
        self.name_ = self.name
        self.all_duration_ = format_time(self.all_duration)
        self.avg_duration_ = format_time(self.avg_duration)
        self.num_called_ = self.num_called
        self.event_type_ = format_event_type(self.event_type)
        self.shapes_ = self.shapes

    def update(self, event):
        self.all_duration += event.all_duration
        self.num_called += event.num_called
        self.avg_duration = self.all_duration / self.num_called
        self.__post_init__()


class Events(list):
    def __init__(self, events: str = "") -> None:
        list.__init__([])
        if events != "":
            self.__init_events(events)

    def __init_events(self, events: str):
        events_json = json.loads(events)
        for event_json in events_json:
            self.append(from_dict(data_class=Event, data=event_json))

    def __str__(self):
        return self.table()

    def key_averages(self):
        stats: Dict[Tuple[str, ...], Event] = OrderedDict()

        def get_key(event: Event) -> Tuple[str, ...]:
            return tuple([event.name_, event.shapes_])

        for event in self:
            key = get_key(event=event)
            if key in stats:
                stats[key].update(event)
            else:
                stats[key] = event
        results = Events()
        results.extend(stats.values())
        return results

    def table(self):
        t = PrettyTable()
        t.field_names = [
            "Name",
            "All duration",
            "Average duration",
            "Number of calls",
            "Event type",
            "Shapes of inputs",
        ]
        for item in self:
            t.add_row(
                [
                    item.name_,
                    item.all_duration_,
                    item.avg_duration_,
                    item.num_called_,
                    item.event_type_,
                    item.shapes_,
                ]
            )
        return t.get_string()
