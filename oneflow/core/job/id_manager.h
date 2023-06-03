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
#ifndef ONEFLOW_CORE_JOB_ID_MANAGER_H_
#define ONEFLOW_CORE_JOB_ID_MANAGER_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/job/id_state.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/resource_desc.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/graph/task_id_generator.h"

namespace oneflow {

class IDMgr final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(IDMgr);
  ~IDMgr() = default;

  int64_t NewRegstDescId();
  int64_t NewMemBlockId();
  int64_t NewChunkId();

  TaskIdGenerator* GetTaskIdGenerator() { return &task_id_gen_; }

  void SaveIdAndTaskIndex(IdState* id_state);
  void TryUpdateIdAndTaskIndex(const IdState* id_state);

 private:
  friend class Singleton<IDMgr>;
  IDMgr();

  std::atomic<int64_t> regst_desc_id_count_;
  std::atomic<int64_t> mem_block_id_count_;
  std::atomic<int64_t> chunk_id_count_;
  TaskIdGenerator task_id_gen_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_ID_MANAGER_H_
