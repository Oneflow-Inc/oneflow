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
#ifdef WITH_CUDA

#include "oneflow/core/vm/cuda_optional_event_record_status_querier.h"
#include "oneflow/core/device/device_context.h"

namespace oneflow {
namespace vm {

CudaOptionalEventRecordStatusQuerier::~CudaOptionalEventRecordStatusQuerier() {
  cuda_event_.reset();
}

bool CudaOptionalEventRecordStatusQuerier::event_completed() const {
  cudaSetDevice(cuda_event_->device_id());
  return cuda_event_->Query();
}

void CudaOptionalEventRecordStatusQuerier::SetLaunched(DeviceCtx* device_ctx) {
  // No lock needed. This function will be called only one time.
  // In most cases, errors will be successfully detected by CHECK
  // even though run in different threads.
  CHECK(!launched_);
  if (cuda_event_) {
    cudaSetDevice(cuda_event_->device_id());
    OF_CUDA_CHECK(cudaEventRecord(*cuda_event_->mut_event(), device_ctx->cuda_stream()));
  }
  launched_ = true;
}

}  // namespace vm
}  // namespace oneflow

#endif
