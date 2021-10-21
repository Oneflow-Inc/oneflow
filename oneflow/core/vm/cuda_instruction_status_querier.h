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
#ifndef ONEFLOW_CORE_VM_CUDA_VM_INSTRUCTION_STATUS_QUERIER_H_
#define ONEFLOW_CORE_VM_CUDA_VM_INSTRUCTION_STATUS_QUERIER_H_

#include "oneflow/core/device/cuda_util.h"

namespace oneflow {

class DeviceCtx;

namespace vm {

#ifdef WITH_CUDA
class CudaInstrStatusQuerier {
 public:
  ~CudaInstrStatusQuerier() = default;

  bool done() const { return launched_ && event_completed(); }
  void SetLaunched(DeviceCtx* device_ctx);

  static const CudaInstrStatusQuerier* Cast(const char* mem_ptr) {
    return reinterpret_cast<const CudaInstrStatusQuerier*>(mem_ptr);
  }
  static CudaInstrStatusQuerier* MutCast(char* mem_ptr) {
    return reinterpret_cast<CudaInstrStatusQuerier*>(mem_ptr);
  }
  static CudaInstrStatusQuerier* PlacementNew(char* mem_ptr, int64_t device_id) {
    return new (mem_ptr) CudaInstrStatusQuerier(device_id);
  }

 private:
  explicit CudaInstrStatusQuerier(int64_t device_id) : launched_(false), device_id_(device_id) {}
  bool event_completed() const;

  volatile bool launched_;
  int64_t device_id_;
  cudaEvent_t event_;
};

#endif

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_CUDA_VM_INSTRUCTION_STATUS_QUERIER_H_
