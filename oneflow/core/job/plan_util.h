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
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/job/plan.pb.h"
#include "oneflow/core/job/job.pb.h"
#include "oneflow/core/graph/stream_id.h"

namespace oneflow {

struct PlanUtil {
  static RegstDescProto* GetSoleProducedDataRegst(TaskProto* task_proto);
  static std::function<const TaskProto*(int64_t)> MakeGetterTaskProto4TaskId(const Plan& plan);
  static void MergeMemBlockIdByLogicalChainId(Plan* plan, const Job& job);
  static void SetUniqueMemBlockId4UnreusedMemRegst(Plan* plan);
  static void GenMemBlockAndChunk4Plan(Plan* plan);
  static void GenMemBlockAndChunkWithVariableOpNames4Plan(
      Plan* plan, const HashSet<std::string>& variable_op_names);
  static void CleanUselessMemBlockAndCheckValid(Plan* plan);
  static void ToDotFile(const Plan& plan, const std::string& filepath);
  static std::function<RegstDescProto*(int64_t)> MakeMutRegstDesc4Id(Plan* plan);
  static void SetForceInplaceMemBlock(Plan* plan);
  static void DumpCtrlRegstInfoToPlan(Plan* plan);
  static void GenCollectiveBoxingPlan(Job* job, Plan* plan);
  static void GenRegisterHint(Plan* plan);
  static void GenLightPlan(Plan* plan, const std::string& plan_name);
  static void PlanMemoryLog(Plan* plan, const std::string& plan_name);
  static const oneflow::OpAttribute& GetOpAttribute(const Plan* plan, int64_t job_id,
                                                    const oneflow::KernelConf& kernel_conf);
  // NOTE(chengcheng): recovery op_attr
  static void PopulateOpAttribute(
      Plan* plan,
      const PbMap<int64_t, ::oneflow::OpAttributeRefTable>& job_id2op_attribute_ref_table);
  static StreamId GetStreamId(const TaskProto& task);
  static int64_t GetDeviceIndex(const TaskProto& task);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_PLAN_UTIL_H_
