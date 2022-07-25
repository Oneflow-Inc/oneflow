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

#ifndef ONEFLOW_CORE_EAGER_DEV_VM_DEP_OBJECT_CONSUME_MODE_H_
#define ONEFLOW_CORE_EAGER_DEV_VM_DEP_OBJECT_CONSUME_MODE_H_

namespace oneflow {
namespace one {

enum class DevVmDepObjectConsumeMode {
  NONE,
  MUTABLE,
};

inline DevVmDepObjectConsumeMode* CurrentDevVmDepObjectConsumeMode() {
  static thread_local DevVmDepObjectConsumeMode mode_ = DevVmDepObjectConsumeMode::MUTABLE;
  return &mode_;
}

class DevVmDepObjectConsumeModeGuard {
 public:
  DevVmDepObjectConsumeModeGuard(DevVmDepObjectConsumeMode mode)
      : prev_mode_(*CurrentDevVmDepObjectConsumeMode()) {
    *CurrentDevVmDepObjectConsumeMode() = mode;
  }
  ~DevVmDepObjectConsumeModeGuard() { *CurrentDevVmDepObjectConsumeMode() = prev_mode_; }  // NOLINT

 private:
  DevVmDepObjectConsumeMode prev_mode_;
};

}  // namespace one
}  // namespace oneflow

#endif  // ONEFLOW_CORE_EAGER_DEV_VM_DEP_OBJECT_CONSUME_MODE_H_
