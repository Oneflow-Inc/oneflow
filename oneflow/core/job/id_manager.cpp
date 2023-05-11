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
#include "oneflow/core/job/id_manager.h"
#include "oneflow/core/framework/multi_client_session_context.h"
#include "oneflow/core/job/id_state.h"

namespace oneflow {

IDMgr::IDMgr() {
  regst_desc_id_count_ = 0;
  mem_block_id_count_ = 0;
  chunk_id_count_ = 0;
}

std::vector<int64_t> IDMgr::GetId() const {
  return {regst_desc_id_count_, mem_block_id_count_, chunk_id_count_};
}

void IDMgr::TryUpdateId(int64_t regst_desc_id_count, int64_t mem_block_id_count,
                        int64_t chunk_id_count) {
  regst_desc_id_count_ = std::max(regst_desc_id_count, regst_desc_id_count_);
  mem_block_id_count_ = std::max(mem_block_id_count, mem_block_id_count_);
  chunk_id_count_ = std::max(chunk_id_count, chunk_id_count_);
}

void IDMgr::GetTaskIndex(HashMap<int64_t, uint32_t>* task_index_state) {
  task_id_gen_.GetTaskIndex(task_index_state);
}

void IDMgr::TryUpdateTaskIndex(const HashMap<int64_t, uint32_t>& task_index_state) {
  task_id_gen_.TryUpdateTaskIndex(task_index_state);
}

}  // namespace oneflow
