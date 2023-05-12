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
#include "oneflow/core/common/singleton.h"
#include "oneflow/core/graph/task_stream_index_manager.h"
#include "oneflow/core/job/id_manager.h"
#include "oneflow/core/job/id_state.h"
#include "oneflow/core/job/job_build_and_infer_ctx_mgr.h"

namespace oneflow {

IdState IdStateMgr::GetIdState() {
  CHECK(Singleton<IDMgr>::Get() != nullptr);
  CHECK(Singleton<TaskStreamIndexManager>::Get() != nullptr);
  CHECK(Singleton<LazyJobBuildAndInferCtxMgr>::Get() != nullptr);

  id_state_.job_id_state_ = Singleton<LazyJobBuildAndInferCtxMgr>::Get()->GetJobIdCount();
  Singleton<IDMgr>::Get()->SaveIdAndTaskIndex(&id_state_);
  Singleton<TaskStreamIndexManager>::Get()->GetTaskStreamIndex(&id_state_.stream_index_state_);
  return id_state_;
}

void IdStateMgr::SetIdState(const IdState& id_state) {
  id_state_ = id_state;
  CHECK(Singleton<IDMgr>::Get() != nullptr);
  CHECK(Singleton<TaskStreamIndexManager>::Get() != nullptr);
  CHECK(Singleton<LazyJobBuildAndInferCtxMgr>::Get() != nullptr);
  Singleton<IDMgr>::Get()->TryUpdateIdAndTaskIndex(&id_state_);
  Singleton<TaskStreamIndexManager>::Get()->TryUpdateTaskStreamIndex(id_state_.stream_index_state_);
  Singleton<LazyJobBuildAndInferCtxMgr>::Get()->TryUpdateJobIdCount(id_state_.job_id_state_);
}

}  // namespace oneflow
