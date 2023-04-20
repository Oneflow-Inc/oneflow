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
#ifndef ONEFLOW_CORE_GRAPH_TASK_GRAPH_REBUILD_CTX_H_
#define ONEFLOW_CORE_GRAPH_TASK_GRAPH_REBUILD_CTX_H_

#include "oneflow/core/common/maybe.h"
#include "oneflow/core/register/register_desc.h"

namespace oneflow {

class TaskNode;
class TaskEdge;

class TaskGraphRebuildCtx {
 public:
  TaskGraphRebuildCtx() = default;
  ~TaskGraphRebuildCtx() = default;

  Maybe<TaskNode*> TaskNode4Id(int64_t task_id) const;
  Maybe<TaskEdge*> TaskEdge4Uid(int64_t task_edge_uid) const;
  Maybe<RegstDesc> RegstDesc4Id(int64_t regst_desc_id) const;

  Maybe<void> AddTaskNode(TaskNode* task_node);
  Maybe<void> AddTaskEdge(TaskEdge* task_edge, int64_t task_edge_uid);
  Maybe<void> AddRegstDesc(const std::shared_ptr<RegstDesc>& regst_desc);

 private:
  HashMap<int64_t, TaskNode*> id2task_node_;
  HashMap<int64_t, TaskEdge*> uid2task_edge_;
  HashMap<int64_t, std::shared_ptr<RegstDesc>> id2regst_desc_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_TASK_GRAPH_REBUILD_CTX_H_
