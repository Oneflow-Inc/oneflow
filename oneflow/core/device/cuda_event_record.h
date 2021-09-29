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
#ifndef ONEFLOW_CORE_DEVICE_CUDA_EVENT_RECORD_H_
#define ONEFLOW_CORE_DEVICE_CUDA_EVENT_RECORD_H_

#include "oneflow/core/device/event_record.h"
#include "oneflow/core/device/cuda_util.h"

namespace oneflow {

#ifdef WITH_CUDA
class DeviceCtx;
class CudaEventRecord final : public EventRecord {
 public:
  CudaEventRecord(const CudaEventRecord&) = delete;
  CudaEventRecord(CudaEventRecord&&) = delete;
  CudaEventRecord& operator=(const CudaEventRecord&) = delete;
  CudaEventRecord& operator=(CudaEventRecord&&) = delete;

  explicit CudaEventRecord(DeviceCtx* device_ctx);
  CudaEventRecord(int64_t device_id, DeviceCtx* device_ctx);
  ~CudaEventRecord() = default;

  bool QueryDone() const override;

 private:
  int64_t device_id_;
  cudaEvent_t event_;
};
#endif

}  // namespace oneflow

#endif  // ONEFLOW_CORE_DEVICE_CUDA_EVENT_RECORD_H_
