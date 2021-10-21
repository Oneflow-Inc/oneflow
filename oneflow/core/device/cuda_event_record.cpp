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
#include "oneflow/core/device/cuda_event_record.h"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/device/device_context.h"
#include "oneflow/core/rpc/include/global_process_ctx.h"

namespace oneflow {

#ifdef WITH_CUDA

namespace {

int GetCurrentDeviceId() {
  int device_id = -1;
  OF_CUDA_CHECK(cudaGetDevice(&device_id));
  CHECK_EQ(device_id, GlobalProcessCtx::LocalRank());
  return device_id;
}

}  // namespace

CudaEventRecord::CudaEventRecord(DeviceCtx* device_ctx)
    : CudaEventRecord(GetCurrentDeviceId(), device_ctx) {}

CudaEventRecord::CudaEventRecord(int64_t device_id, DeviceCtx* device_ctx) : device_id_(device_id) {
  CudaCurrentDeviceGuard guard(device_id_);
  OF_CUDA_CHECK(cudaEventCreateWithFlags(&event_, cudaEventBlockingSync | cudaEventDisableTiming));
  OF_CUDA_CHECK(cudaEventRecord(event_, device_ctx->cuda_stream()));
}

bool CudaEventRecord::QueryDone() const {
  CudaCurrentDeviceGuard guard(device_id_);
  return cudaEventQuery(event_) != cudaErrorNotReady;
}

#endif  // WITH_CUDA

}  // namespace oneflow
