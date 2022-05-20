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
#include "oneflow/core/ep/hip/cuda_event.h"

#ifdef WITH_ROCM

namespace oneflow {

namespace ep {

CudaEvent::CudaEvent(unsigned int flags) : cuda_event_{} {
  OF_CUDA_CHECK(hipEventCreateWithFlags(&cuda_event_, flags));
}

CudaEvent::~CudaEvent() { OF_CUDA_CHECK(hipEventDestroy(cuda_event_)); }

Maybe<bool> CudaEvent::QueryDone() {
  hipError_t err = hipEventQuery(cuda_event_);
  if (err == hipSuccess) {
    return Maybe<bool>(true);
  } else if (err == hipErrorNotReady) {
    return Maybe<bool>(false);
  } else {
    return Error::RuntimeError() << hipGetErrorString(err);
  }
}

Maybe<void> CudaEvent::Sync() {
  hipError_t err = hipEventSynchronize(cuda_event_);
  if (err == hipSuccess) {
    return Maybe<void>::Ok();
  } else {
    return Error::RuntimeError() << hipGetErrorString(err);
  }
}

hipEvent_t CudaEvent::cuda_event() { return cuda_event_; }

}  // namespace ep

}  // namespace oneflow

#endif  // WITH_ROCM
