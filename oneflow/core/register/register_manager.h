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
#ifndef ONEFLOW_CORE_REGISTER_REGISTER_MANAGER_H_
#define ONEFLOW_CORE_REGISTER_REGISTER_MANAGER_H_

#include <mutex>

#include "oneflow/core/job/id_manager.h"
#include "oneflow/core/job/plan.pb.h"
#include "oneflow/core/job/runtime_context.h"
#include "oneflow/core/memory/memory_allocator.h"
#include "oneflow/core/register/blob.h"
#include "oneflow/core/register/logical_blob_id.pb.h"
#include "oneflow/core/register/register.h"
#include "oneflow/core/record/record.pb.h"

namespace oneflow {

namespace vm {
class EagerBlobObject;
}

class RegstMgr final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RegstMgr);
  RegstMgr();
  ~RegstMgr() = default;

  void AddPlan(
      const Plan& plan,
      const HashMap<std::string, vm::EagerBlobObject*>& variable_op_name2eager_blob_object);
  void AddPlan(const Plan& plan);
  void NewRegsts(const RegstDescProto& regst_desc_proto, std::function<void(Regst*)> OneRegstDone);
  const RtRegstDesc& RegstDesc4RegstDescId(int64_t regst_desc_id) const;
  bool HasRegstDescId(int64_t regst_desc_id) const;
  int64_t ProducerTaskId4RegstDescId(int64_t regst_desc_id) const;
  bool HasProducerTaskId4RegstDescId(int64_t regst_desc_id) const;

 private:
  bool IsStreamOrderedMemoryAllocationCase(const MemoryCase& mem_case) const;

  HashMap<int64_t, std::unique_ptr<const RtRegstDesc>> regst_desc_id2rt_regst_desc_;
  HashMap<int64_t, char*> mem_block_id2ptr_;
  HashSet<int64_t> stream_ordered_allocation_mem_block_ids_;
  HashMap<int64_t, int64_t> ctrl_regst_desc_id2producer_task_id_;
  std::mutex mutex_;
  bool stream_ordered_memory_allocation_enabled_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_REGISTER_REGISTER_MANAGER_H_
