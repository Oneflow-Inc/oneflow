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
from oneflow.profiler.events import Event, Events


class TestEventAndEvents(flow.unittest.TestCase):
    def test_event(test_case):
        event = Event("test", 1234, False, -1, 1, "-", 0)
        event_json = {
            "name": "test",
            "time": 1234,
            "on_gpu": False,
            "input_shapes": "-",
            "type": 0,
        }
        test_case.assertEqual(event, Event.from_dict(event_json))

        event1 = Event("test", 3346, False, -1, 1, "-", 0)
        event.update(event1)
        test_case.assertEqual(event.count, 2)
        test_case.assertEqual(event.time, 2290)
        test_case.assertEqual(event.time_total, 4580)
        test_case.assertEqual(event.on_gpu, False)

    def test_events(test_case):
        events_json = json.dumps(
            [
                {
                    "name": "test",
                    "time": 1234,
                    "on_gpu": False,
                    "input_shapes": "-",
                    "type": 0,
                },
                {
                    "name": "test",
                    "time": 3346,
                    "on_gpu": False,
                    "input_shapes": "-",
                    "type": 0,
                },
            ]
        )
        events = [
            Event("test", 1234, False, -1, 1, "-", 0),
            Event("test", 3346, False, -1, 1, "-", 0),
        ]
        events_avg = [Event("test", 2290, False, -1, 2, "-", 0)]
        test_case.assertEqual(Events(events_json), events)
        test_case.assertEqual(Events(events_json).key_averages(), events_avg)


if __name__ == "__main__":
    unittest.main()
