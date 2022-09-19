#include "oneflow/core/common/container_util.h"
#include "oneflow/core/graph/task_node.h"
#include "oneflow/core/graph/task_graph_rebuild_ctx.h"

namespace oneflow {

Maybe<TaskNode*> TaskGraphRebuildCtx::TaskNode4Id(int64_t task_id) const {
  return JUST(MapAt(id2task_node_, task_id));
}

Maybe<TaskEdge*> TaskGraphRebuildCtx::TaskEdge4Uid(int64_t task_edge_uid) const {
  return JUST(MapAt(uid2task_edge_, task_edge_uid));
}

Maybe<RegstDesc> TaskGraphRebuildCtx::RegstDesc4Id(int64_t regst_desc_id) {
  return JUST(MapAt(id2regst_desc_, regst_desc_id));
}

Maybe<void> TaskGraphRebuildCtx::AddTaskNode(TaskNode* task_node) {
  CHECK_OR_RETURN(id2task_node_.emplace(task_node->task_id(), task_node).second);
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
 
}
