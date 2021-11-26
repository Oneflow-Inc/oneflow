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
#ifndef ONEFLOW_CORE_EP_CUDA_CUDA_EVENT_H_
#define ONEFLOW_CORE_EP_CUDA_CUDA_EVENT_H_

#include "oneflow/core/ep/include/event.h"

#ifdef WITH_CUDA

#include "oneflow/core/device/cuda_util.h"

namespace oneflow {

namespace ep {

class CudaEvent : public Event {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CudaEvent);
  explicit CudaEvent(unsigned int flags);
  ~CudaEvent() override;

  Maybe<bool> QueryDone() override;
  Maybe<void> Sync() override;

  cudaEvent_t cuda_event();

 private:
  cudaEvent_t cuda_event_;
};

}  // namespace ep

}  // namespace oneflow

#endif  // WITH_CUDA

#endif  // ONEFLOW_CORE_EP_CUDA_CUDA_EVENT_H_
