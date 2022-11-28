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
#ifndef ONEFLOW_CORE_FRAMEWORK_STREAM_GET_STREAM_TYPE_NAME_H_
#define ONEFLOW_CORE_FRAMEWORK_STREAM_GET_STREAM_TYPE_NAME_H_

#include <glog/logging.h>
#include <string>
#include "oneflow/core/common/stream_type.h"
#include "oneflow/core/common/device_type.h"
#include "oneflow/core/framework/to_string.h"

namespace oneflow {

struct GetStreamTypeName : public StreamTypeVisitor<GetStreamTypeName> {
  static const char* VisitCompute() { return "compute"; }
  static const char* VisitHost2Device() { return "h2d"; }
  static const char* VisitDevice2Host() { return "d2h"; }
  static const char* VisitCcl() { return "ccl"; }
  static const char* VisitBarrier() { return "barrier"; }
  static const char* VisitCriticalSection() { return "critical_section"; }
  static const char* VisitLazyJobLauncher() { return "lazy_job_launcher"; }
  static const char* VisitPinnedCompute() { return "pinned_compute"; }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_STREAM_GET_STREAM_TYPE_NAME_H_
