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
#include <vector>
#include "oneflow/core/device/cuda_event.h"

namespace oneflow {

#ifdef WITH_CUDA

CudaEvent::CudaEvent(int device_id, unsigned int flags) : device_id_(device_id) {
  CudaCurrentDeviceGuard guard(device_id_);
  OF_CUDA_CHECK(cudaEventCreateWithFlags(&event_, flags));
}

CudaEvent::~CudaEvent() {
  CudaCurrentDeviceGuard guard(device_id_);
  OF_CUDA_CHECK(cudaEventDestroy(event_));
}

bool CudaEvent::Query() const { return cudaEventQuery(event_) != cudaErrorNotReady; }

#endif

}  // namespace oneflow
