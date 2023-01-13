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
#ifndef ONEFLOW_CORE_VM_STREAM_GET_ALLOCATOR_STREAM_TYPE_H_
#define ONEFLOW_CORE_VM_STREAM_GET_ALLOCATOR_STREAM_TYPE_H_

#include "oneflow/core/common/stream_type.h"

namespace oneflow {

struct GetAllocatorStreamType final : public StreamTypeVisitor<GetAllocatorStreamType> {
  static Maybe<StreamType> VisitCompute() { return StreamType::kCompute; }
  static Maybe<StreamType> VisitHost2Device() { return StreamType::kCompute; }
  static Maybe<StreamType> VisitCcl() { return StreamType::kCompute; }
  static Maybe<StreamType> VisitPinnedCompute() { return StreamType::kPinnedCompute; }
  static Maybe<StreamType> VisitDevice2Host() { return StreamType::kDevice2Host; }
  static Maybe<StreamType> VisitBarrier() {
    UNIMPLEMENTED_THEN_RETURN() << "no allocator supported on 'barrier' stream_type.";
  }
  static Maybe<StreamType> VisitCriticalSection() {
    UNIMPLEMENTED_THEN_RETURN() << "no allocator supported on 'critical_section' stream_type.";
  }
  static Maybe<StreamType> VisitLazyJobLauncher() {
    UNIMPLEMENTED_THEN_RETURN() << "no allocator supported on 'lazy_job_launcher' stream_type.";
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_STREAM_GET_ALLOCATOR_STREAM_TYPE_H_
