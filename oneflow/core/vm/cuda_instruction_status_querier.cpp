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

#include "oneflow/core/vm/cuda_instruction_status_querier.h"
#include "oneflow/core/device/device_context.h"

namespace oneflow {
namespace vm {

bool CudaInstrStatusQuerier::event_completed() const {
  cudaSetDevice(device_id_);
  return cudaEventQuery(event_) == cudaSuccess;
}

void CudaInstrStatusQuerier::SetLaunched(DeviceCtx* device_ctx) {
  cudaSetDevice(device_id_);
  OF_CUDA_CHECK(cudaEventCreateWithFlags(&event_, cudaEventBlockingSync | cudaEventDisableTiming));
  OF_CUDA_CHECK(cudaEventRecord(event_, device_ctx->cuda_stream()));
  launched_ = true;
}

}  // namespace vm
}  // namespace oneflow

#endif
