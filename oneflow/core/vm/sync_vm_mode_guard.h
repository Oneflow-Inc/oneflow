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
#ifndef ONEFLOW_CORE_VM_SYNC_VM_MODE_GUARD_H_
#define ONEFLOW_CORE_VM_SYNC_VM_MODE_GUARD_H_

#include "oneflow/core/common/thread_local_guard.h"

namespace oneflow {

enum class SyncVmMode {
  kInvalid = 0,
  kEnable = 1,
  kDisable = 2,
};

class SyncVmModeGuard final : public ThreadLocalGuard<SyncVmMode> {
 public:
  using ThreadLocalGuard<SyncVmMode>::ThreadLocalGuard;
  ~SyncVmModeGuard() = default;

  static bool IsCurrentSyncVmMode() {
    const auto& opt_sync_mode = Current();
    return opt_sync_mode.has_value() && CHECK_JUST(opt_sync_mode) == SyncVmMode::kEnable;
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_SYNC_VM_MODE_GUARD_H_
