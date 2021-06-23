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
#ifndef ONEFLOW_CORE_JOB_IMPROVER_H_
#define ONEFLOW_CORE_JOB_IMPROVER_H_

#include "oneflow/core/job/plan.pb.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/memory/memory_case.pb.h"
#include "oneflow/core/job/available_memory_desc.pb.h"
#include "oneflow/core/graph/chain_act_graph.h"

namespace oneflow {

class Improver final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Improver);
  Improver() : start_mem_block_id_(-1) {}
  ~Improver() = default;

 private:
  void Init(const AvailableMemDesc& amd, const Plan& naive_plan);
  Maybe<void> ForEachImprovedRegstNum(
      const Plan& plan, bool is_memory_limited, double ii,
      const std::function<const HashMap<int64_t, double>&(int64_t)>& PathDurations4RegstDescId,
      const std::function<const HashMap<int64_t, double>&(int64_t)>& PathIIScales4RegstDescId,
      const std::function<void(int64_t, uint64_t)>& Handler) const;
  //  first dimension index of MemZoneRegstDescs is machine_id
  //  second dimension index of MemZoneRegstDescs is mem_zone_id
  using MemZoneRegstDescs = std::vector<std::vector<std::list<const RegstDescProto*>>>;
  Maybe<void> CheckAllZoneNotOOM(
      const MemZoneRegstDescs& mz_regst_descs,
      const std::function<const HashMap<int64_t, double>&(int64_t)>& Duration4RegstDescId,
      const std::function<const HashMap<int64_t, double>&(int64_t)>& Ratio4RegstDescId,
      double ii) const;
  Maybe<double> BinarySearchII(
      double base_ii,
      const std::function<const HashMap<int64_t, double>&(int64_t)>& Duration4RegstDescId,
      const std::function<const HashMap<int64_t, double>&(int64_t)>& Ratio4RegstDescId,
      const MemZoneRegstDescs& mz_regst_descs) const;
  uint64_t AvailableMemSize(int64_t machine_id, int64_t memory_zone_id) const;
  int64_t GetMemoryZoneId(const MemoryCase& mem_case) const;
  void MakeMemZoneRegstDescs(const Plan& plan, MemZoneRegstDescs* mz2regst_desc) const;
  double CalcMaxRegstDescDuration(
      const std::function<const HashMap<int64_t, double>&(int64_t)>& Duration4RegstDescId,
      const MemZoneRegstDescs& mz_regst_descs) const;

  int32_t start_mem_block_id_;
  AvailableMemDesc amd_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_IMPROVER_H_
