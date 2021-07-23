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
#ifndef ONEFLOW_CORE_JOB_IN_JOB_MEM_SHARING_UTIL_H_
#define ONEFLOW_CORE_JOB_IN_JOB_MEM_SHARING_UTIL_H_

#include "oneflow/core/job/plan.pb.h"
#include <functional>
#include <string>

namespace oneflow {

struct IntraJobMemSharingUtil {
  static void InferMemBlockId4MemReusedRegst(
      Plan* plan, const std::function<bool(const std::string&, const std::string&)>&
                      IsOpNameDataOrCtrlReachable);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_IN_JOB_MEM_SHARING_UTIL_H_
