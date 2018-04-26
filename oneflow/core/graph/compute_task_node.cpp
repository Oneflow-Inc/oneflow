#include "oneflow/core/graph/compute_task_node.h"
#include "oneflow/core/graph/task_graph.h"
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

namespace {

const LogicalNode* LogicalNodeOnEdge(TaskEdge* edge, TaskNode* (TaskEdge::*GetNode)() const,
                                     const std::unordered_set<TaskEdge*>& (TaskNode::*GetEdges)()
                                         const) {
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
  if (target_node) { return target_node->logical_node(); }
  return nullptr;
}

}  // namespace

void CompTaskNode::ToProto(TaskProto* task_proto) {
  TaskNode::ToProto(task_proto);
  *(task_proto->mutable_parallel_ctx()) = parallel_ctx_;
}

const LogicalNode* CompTaskNode::GetOneSuccLogicalNodeOnEdge(TaskEdge* edge) {
  return LogicalNodeOnEdge(edge, &TaskEdge::dst_node, &TaskNode::out_edges);
}

const LogicalNode* CompTaskNode::GetOnePredLogicalNodeOnEdge(TaskEdge* edge) {
  return LogicalNodeOnEdge(edge, &TaskEdge::src_node, &TaskNode::in_edges);
}

void CompTaskNode::BindEdgeWithProducedBoxingOr121Regst(TaskEdge* out_edge,
                                                        const std::string& boxing_regst_name,
                                                        const std::string& one2one_regst_name) {
  BindEdgeWithProducedBoxingOr121Regst(out_edge, GetOneSuccLogicalNodeOnEdge(out_edge),
                                       boxing_regst_name, one2one_regst_name);
}

void CompTaskNode::BindEdgeWithProducedBoxingOr121Regst(TaskEdge* out_edge,
                                                        const LogicalNode* succ_node,
                                                        const std::string& boxing_regst_name,
                                                        const std::string& one2one_regst_name) {
  BldSubTskGphMthd mthd = GetMthdForBldSubTskGph(logical_node(), succ_node);
  if (mthd == &TaskGraph::BldSubTskGphByBoxing) {
    BindEdgeWithProducedRegst(out_edge, boxing_regst_name);
  } else if (mthd == &TaskGraph::BldSubTskGphByOneToOne) {
    BindEdgeWithProducedRegst(out_edge, one2one_regst_name);
  } else {
    UNIMPLEMENTED();
  }
}

}  // namespace oneflow
