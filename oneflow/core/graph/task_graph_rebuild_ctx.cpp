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
#include "oneflow/core/common/container_util.h"
#include "oneflow/core/graph/task_node.h"
#include "oneflow/core/graph/task_graph_rebuild_ctx.h"

namespace oneflow {

Maybe<TaskNode*> TaskGraphRebuildCtx::TaskNode4Id(int64_t task_id) const {
  auto* task_node = JUST(MapAt(id2task_node_, task_id));
  CHECK_EQ_OR_RETURN(task_node->task_id(), task_id);  // NOLINT
  return task_node;
}

Maybe<TaskEdge*> TaskGraphRebuildCtx::TaskEdge4Uid(int64_t task_edge_uid) const {
  return JUST(MapAt(uid2task_edge_, task_edge_uid));
}

Maybe<RegstDesc> TaskGraphRebuildCtx::RegstDesc4Id(int64_t regst_desc_id) const {
  return JUST(MapAt(id2regst_desc_, regst_desc_id));
}

Maybe<void> TaskGraphRebuildCtx::AddTaskNode(TaskNode* task_node) {
  CHECK_OR_RETURN(id2task_node_.emplace(task_node->task_id(), task_node).second)
      << "redundant task id found. value: " << task_node->task_id();
  for (const auto& pair : task_node->produced_regsts()) { JUST(AddRegstDesc(pair.second)); }
  return Maybe<void>::Ok();
}

Maybe<void> TaskGraphRebuildCtx::AddTaskEdge(TaskEdge* task_edge, int64_t task_edge_uid) {
  CHECK_OR_RETURN(uid2task_edge_.emplace(task_edge_uid, task_edge).second)
      << "redundant task edge uid found. value: " << task_edge_uid;
  return Maybe<void>::Ok();
}

Maybe<void> TaskGraphRebuildCtx::AddRegstDesc(const std::shared_ptr<RegstDesc>& regst_desc) {
  CHECK_OR_RETURN(id2regst_desc_.emplace(regst_desc->regst_desc_id(), regst_desc).second)
      << "redundant register descriptor id found. value: " << regst_desc->regst_desc_id();
  return Maybe<void>::Ok();
}

}  // namespace oneflow
