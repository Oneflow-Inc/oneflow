#include "oneflow/core/graph/compute_task_node.h"
#include "oneflow/core/graph/logical_node.h"
#include "oneflow/core/graph/task_graph.h"

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

void CompTaskNode::ProduceB121Regst(const std::string& name) {
  ProduceRegst("boxing_" + name);
  ProduceRegst("121_" + name);
}

void CompTaskNode::BindEdgeWithProducedB121Regst(TaskEdge* edge, const std::string& b121_name) {
  BldSubTskGphMthd mthd = GetMthdForBldSubTskGph(logical_node(), GetOneSuccLogicalNodeOnEdge(edge));
  if (mthd == &TaskGraph::BldSubTskGphByBoxing) {
    BindEdgeWithProducedRegst(edge, "boxing_" + b121_name);
  } else if (mthd == &TaskGraph::BldSubTskGphByOneToOne) {
    BindEdgeWithProducedRegst(edge, "121_" + b121_name);
  } else {
    UNIMPLEMENTED();
  }
}

bool CompTaskNode::TryAddLbiToB121RegstAndBindIt(ExecNode* exec_node, const std::string& bn,
                                                 const std::string& b121_name) {
  std::shared_ptr<RegstDesc> regst_boxing = GetProducedRegst("boxing_" + b121_name);
  std::shared_ptr<RegstDesc> regst_121 = GetProducedRegst("121_" + b121_name);
  const HashSet<LogicalBlobId>& lbi_boxing = logical_node()->lbi_boxing();
  const HashSet<LogicalBlobId>& lbi_121 = logical_node()->lbi_121();
  const LogicalBlobId& lbi = exec_node->op()->BnInOp2Lbi(bn);
  if (lbi_boxing.find(lbi) != lbi_boxing.end()) {
    regst_boxing->AddLbi(lbi);
    exec_node->BindBnWithRegst(bn, regst_boxing);
  } else if (lbi_121.find(lbi) != lbi_121.end()) {
    regst_121->AddLbi(lbi);
    exec_node->BindBnWithRegst(bn, regst_121);
  } else {
    return false;
  }
  return true;
}

}  // namespace oneflow
