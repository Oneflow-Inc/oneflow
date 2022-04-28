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
#ifndef ONEFLOW_CORE_VM_NPU_OPTIONAL_EVENT_RECORD_STATUS_QUERIER_H_
#define ONEFLOW_CORE_VM_NPU_OPTIONAL_EVENT_RECORD_STATUS_QUERIER_H_

#include <atomic>
#include "oneflow/core/device/npu_util.h"
#include "oneflow/core/device/npu_event.h"

namespace oneflow {

class DeviceCtx;

namespace vm {

#ifdef WITH_NPU

class NpuOptionalEventRecordStatusQuerier {
 public:
  ~NpuOptionalEventRecordStatusQuerier();

  bool done() const { return launched_ && (!npu_event_ || event_completed()); }
  void SetLaunched(DeviceCtx* device_ctx);

  void reset_npu_event(const std::shared_ptr<NpuEvent>& npu_event) { npu_event_ = npu_event; }

  static const NpuOptionalEventRecordStatusQuerier* Cast(const char* mem_ptr) {
    return reinterpret_cast<const NpuOptionalEventRecordStatusQuerier*>(mem_ptr);
  }
  static NpuOptionalEventRecordStatusQuerier* MutCast(char* mem_ptr) {
    return reinterpret_cast<NpuOptionalEventRecordStatusQuerier*>(mem_ptr);
  }
  static NpuOptionalEventRecordStatusQuerier* PlacementNew(
      char* mem_ptr, const std::shared_ptr<NpuEvent>& npu_event) {
    return new (mem_ptr) NpuOptionalEventRecordStatusQuerier(npu_event);
  }

 private:
  explicit NpuOptionalEventRecordStatusQuerier(const std::shared_ptr<NpuEvent>& npu_event)
      : launched_(false), npu_event_(npu_event) {}
  bool event_completed() const;

  std::atomic<bool> launched_;
  std::shared_ptr<NpuEvent> npu_event_;
};

#endif

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_NPU_OPTIONAL_EVENT_RECORD_STATUS_QUERIER_H_
