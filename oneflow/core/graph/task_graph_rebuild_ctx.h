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

}

#endif  // ONEFLOW_CORE_GRAPH_TASK_GRAPH_REBUILD_CTX_H_
