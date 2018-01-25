#include "oneflow/core/graph/compute_task_node.h"
#include "oneflow/core/graph/chain_node.h"

namespace oneflow {

namespace {

const ChainNode* ChainNodeOnEdge(
    TaskEdge* edge, TaskNode* (TaskEdge::*GetNode)() const,
    const std::unordered_set<TaskEdge*>& (TaskNode::*GetEdges)() const) {
  CompTaskNode* target_node = nullptr;
  do {
    TaskNode* tmp_node = (edge->*GetNode)();
    target_node = dynamic_cast<CompTaskNode*>(tmp_node);
    const HashSet<TaskEdge*>& edges = (tmp_node->*GetEdges)();
    if (edges.size() > 0) {
      edge = *(edges.begin());
    } else {
      edge = nullptr;
    }
  } while (!target_node && edge);
  if (target_node) { return target_node->chain_node(); }
  return nullptr;
}

}  // namespace

void CompTaskNode::ToProto(TaskProto* task_proto) {
  TaskNode::ToProto(task_proto);
  *(task_proto->mutable_parallel_ctx()) = parallel_ctx_;
}

const ChainNode* CompTaskNode::SuccChainNodeOnEdge(TaskEdge* edge) {
  return ChainNodeOnEdge(edge, &TaskEdge::dst_node, &TaskNode::out_edges);
}

const ChainNode* CompTaskNode::PredChainNodeOnEdge(TaskEdge* edge) {
  return ChainNodeOnEdge(edge, &TaskEdge::src_node, &TaskNode::in_edges);
}

void SortByParallelId(std::vector<CompTaskNode*>* node_vec) {
  std::sort(node_vec->begin(), node_vec->end(),
            [](const CompTaskNode* lhs, const CompTaskNode* rhs) {
              return lhs->parallel_ctx()->parallel_id()
                     < rhs->parallel_ctx()->parallel_id();
            });
}

}  // namespace oneflow
