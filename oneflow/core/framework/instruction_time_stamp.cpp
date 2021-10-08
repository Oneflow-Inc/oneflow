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
#include "oneflow/core/framework/instruction_time_stamp.h"

#include "oneflow/core/profiler/profiler.h"

namespace oneflow {

namespace {

bool* GetThreadLocalMode() {
  static thread_local bool mode = false;
  return &mode;
}

}  // namespace

bool RecordingInstructionTimeStamp() { return *GetThreadLocalMode(); }

void RecordInstructionTimeStamp(ObjectMsgPtr<vm::InstructionMsg>& instruction) {
  instruction->set_time_stamp(profiler::nanos());
}

RecordInstructionTimeStampMode::RecordInstructionTimeStampMode(bool mode)
    : prev_mode_(*GetThreadLocalMode()) {
  *GetThreadLocalMode() = mode;
}

RecordInstructionTimeStampMode::~RecordInstructionTimeStampMode() {
  *GetThreadLocalMode() = prev_mode_;
}

}  // namespace oneflow
