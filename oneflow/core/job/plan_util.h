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
#ifndef ONEFLOW_CORE_JOB_PLAN_UTIL_H_
#define ONEFLOW_CORE_JOB_PLAN_UTIL_H_

#include <functional>
#include "oneflow/core/job/plan.pb.h"

namespace oneflow {

struct PlanUtil {
  static RegstDescProto* GetSoleProducedDataRegst(TaskProto* task_proto);
  static std::function<const TaskProto*(int64_t)> MakeGetterTaskProto4TaskId(const Plan& plan);
  static void SetUniqueMemBlockId4UnreusedMemRegst(Plan* plan);
  static void GenMemBlockAndChunk4Plan(Plan* plan);
  static void CleanUselessMemBlockAndCheckValid(Plan* plan);
  static void ToDotFile(const Plan& plan, const std::string& filepath);
  static std::function<RegstDescProto*(int64_t)> MakeMutRegstDesc4Id(Plan* plan);
  static void SetForceInplaceMemBlock(Plan* plan);
  static const oneflow::OpAttribute& GeOpAttribute(const Plan* plan, int64_t job_id,
                                                   const oneflow::KernelConf& kernel_conf);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_PLAN_UTIL_H_
