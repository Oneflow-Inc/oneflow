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

void IDMgr::SaveIdAndTaskIndex(IdState* id_state) {
  id_state->regst_desc_id_state_ = regst_desc_id_count_;
  id_state->mem_block_id_state_ = mem_block_id_count_;
  id_state->chunk_id_state_ = chunk_id_count_;
  task_id_gen_.GetTaskIndex(&id_state->task_index_state_);
}

void IDMgr::TryUpdateIdAndTaskIndex(const IdState* id_state) {
  regst_desc_id_count_ = std::max(regst_desc_id_count_, id_state->regst_desc_id_state_);
  mem_block_id_count_ = std::max(mem_block_id_count_, id_state->mem_block_id_state_);
  chunk_id_count_ = std::max(chunk_id_count_, id_state->chunk_id_state_);
  task_id_gen_.TryUpdateTaskIndex(id_state->task_index_state_);
}

}  // namespace oneflow
