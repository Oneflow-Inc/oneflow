/*
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
*/

#include <pybind11/pybind11.h>
#include "oneflow/api/python/of_api_registry.h"
#include "oneflow/core/framework/instruction_replay.h"

namespace py = pybind11;

namespace oneflow {

namespace debug {

ONEFLOW_API_PYBIND11_MODULE("debug", m) {
  m.def("start_recording_instructions", &StartRecordingInstructions);
  m.def("end_recording_instructions", &EndRecordingInstructions);
  m.def("clear_recorded_instructions", &ClearRecordedInstructions);
  m.def("replay_instructions", &ReplayInstructions);
}

}  // namespace debug

}  // namespace oneflow
