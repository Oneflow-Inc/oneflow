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

class RegstMgr final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RegstMgr);
  RegstMgr() = delete;
  ~RegstMgr() = default;

  void NewRegsts(const RegstDescProto& regst_desc_proto, std::function<void(Regst*)> OneRegstDone);
  const RtRegstDesc& RegstDesc4RegstDescId(int64_t regst_desc_id) const;
  bool HasRegstDescId(int64_t regst_desc_id) const;
  int64_t ProducerTaskId4RegstDescId(int64_t regst_desc_id) const;
  bool HasProducerTaskId4RegstDescId(int64_t regst_desc_id) const;
  Blob* Blob4LbiAndParallelId(const LogicalBlobId& lbi, const int64_t parallel_id);

 private:
  friend class Global<RegstMgr>;

  explicit RegstMgr(const Plan& plan);
  void NewBlobsInOneRegst(const std::vector<LbiBlobDescPair>& lbis, Regst*, const RtRegstDesc*,
                          char* main_mem_ptr, char* separated_header_mem_ptr);
  HashMap<int64_t, std::unique_ptr<const RtRegstDesc>> regst_desc_id2rt_regst_desc_;
  HashMap<LogicalBlobId, HashMap<int64_t, Blob*>> lbi2parallel_id2blob_;
  HashMap<int64_t, char*> mem_block_id2ptr_;
  HashMap<int64_t, ParallelContext> regst_desc_id2parallel_ctx_;
  HashMap<int64_t, int64_t> ctrl_regst_desc_id2producer_task_id_;
  std::mutex mutex_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_REGISTER_REGISTER_MANAGER_H_
