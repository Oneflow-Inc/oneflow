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
import unittest
import oneflow.unittest
import oneflow as flow
from oneflow.profiler.events import *


class TestEventAndEvents(flow.unittest.TestCase):
    def test_event(test_case):
        classes = [CustomEvent, KernelEvent]
        custom_event = CustomEvent("custom", 1234, CustomEventType.Default)
        custom_event_json = {
            "name": "custom",
            "time": 1234,
            "custom_type": 0,
            "type": 0,
        }
        test_case.assertEqual(
            custom_event,
            classes[custom_event_json.get("type")].from_dict(custom_event_json),
        )

        kernel_event = KernelEvent("kernel", 1234, 1024, "-")
        kernel_event_json = {
            "name": "kernel",
            "time": 1234,
            "memory_size": 1024,
            "type": 1,
            "input_shapes": "-",
        }
        test_case.assertEqual(
            kernel_event,
            classes[kernel_event_json.get("type")].from_dict(kernel_event_json),
        )

    def test_event_update(test_case):
        event = CustomEvent("custom", 1234, CustomEventType.Default)
        event1 = CustomEvent("custom", 3346, CustomEventType.Default)
        event.update(event1)
        test_case.assertEqual(event.count, 2)
        test_case.assertEqual(event.cpu_time, 2290)
        test_case.assertEqual(event.cpu_time_total, 4580)

    def test_events(test_case):
        events_json = json.dumps(
            [
                {"name": "custom", "time": 1234, "custom_type": 0, "type": 0},
                {"name": "custom", "time": 3346, "custom_type": 0, "type": 0},
            ]
        )
        events = [
            CustomEvent("custom", 1234, CustomEventType.Default),
            CustomEvent("custom", 3346, CustomEventType.Default),
        ]
        events_avg = [CustomEvent("custom", 4580, CustomEventType.Default)]
        events_avg[0].count = 2
        test_case.assertEqual(Events(events_json), events)
        test_case.assertEqual(Events(events_json).key_averages(), events_avg)


if __name__ == "__main__":
    unittest.main()
