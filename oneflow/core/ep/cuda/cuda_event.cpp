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
#include "oneflow/core/ep/cuda/cuda_event.h"

#ifdef WITH_CUDA

namespace oneflow {

namespace ep {

CudaEvent::CudaEvent(unsigned int flags) : cuda_event_{} {
  OF_CUDA_CHECK(cudaEventCreateWithFlags(&cuda_event_, flags));
}

CudaEvent::~CudaEvent() { OF_CUDA_CHECK(cudaEventDestroy(cuda_event_)); }

Maybe<bool> CudaEvent::QueryDone() {
  cudaError_t err = cudaEventQuery(cuda_event_);
  if (err == cudaSuccess) {
    return Maybe<bool>(true);
  } else if (err == cudaErrorNotReady) {
    return Maybe<bool>(false);
  } else {
    return Error::RuntimeError() << cudaGetErrorString(err);
  }
}

Maybe<void> CudaEvent::Sync() {
  cudaError_t err = cudaEventSynchronize(cuda_event_);
  if (err == cudaSuccess) {
    return Maybe<void>::Ok();
  } else {
    return Error::RuntimeError() << cudaGetErrorString(err);
  }
}

cudaEvent_t CudaEvent::cuda_event() { return cuda_event_; }

}  // namespace ep

}  // namespace oneflow

#endif  // WITH_CUDA
