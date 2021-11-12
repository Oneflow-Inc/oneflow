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
#ifndef ONEFLOW_CORE_VM_CUDA_OPTIONAL_EVENT_RECORD_STATUS_QUERIER_H_
#define ONEFLOW_CORE_VM_CUDA_OPTIONAL_EVENT_RECORD_STATUS_QUERIER_H_

#include <atomic>
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/device/cuda_event.h"

namespace oneflow {

class DeviceCtx;

namespace vm {

#ifdef WITH_CUDA

class CudaOptionalEventRecordStatusQuerier {
 public:
  ~CudaOptionalEventRecordStatusQuerier();

  bool done() const { return launched_ && (!cuda_event_ || event_completed()); }
  void SetLaunched(DeviceCtx* device_ctx);

  void reset_cuda_event(const std::shared_ptr<CudaEvent>& cuda_event) { cuda_event_ = cuda_event; }

  static const CudaOptionalEventRecordStatusQuerier* Cast(const char* mem_ptr) {
    return reinterpret_cast<const CudaOptionalEventRecordStatusQuerier*>(mem_ptr);
  }
  static CudaOptionalEventRecordStatusQuerier* MutCast(char* mem_ptr) {
    return reinterpret_cast<CudaOptionalEventRecordStatusQuerier*>(mem_ptr);
  }
  static CudaOptionalEventRecordStatusQuerier* PlacementNew(
      char* mem_ptr, const std::shared_ptr<CudaEvent>& cuda_event) {
    return new (mem_ptr) CudaOptionalEventRecordStatusQuerier(cuda_event);
  }

 private:
  explicit CudaOptionalEventRecordStatusQuerier(const std::shared_ptr<CudaEvent>& cuda_event)
      : launched_(false), cuda_event_(cuda_event) {}
  bool event_completed() const;

  std::atomic<bool> launched_;
  std::shared_ptr<CudaEvent> cuda_event_;
};

#endif

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_CUDA_OPTIONAL_EVENT_RECORD_STATUS_QUERIER_H_
