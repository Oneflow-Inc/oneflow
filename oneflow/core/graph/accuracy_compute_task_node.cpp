#include "oneflow/core/graph/accuracy_compute_task_node.h"
#include "oneflow/core/graph/task_graph.h"
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

void AccuracyCompTaskNode::ProduceAllRegstsAndBindEdges() {
  ProduceRegst("accuracy", false);
  for (TaskEdge* edge : out_edges()) { BindEdgeWithProducedRegst(edge, "accuracy"); }
}

void AccuracyCompTaskNode::ConsumeAllRegsts() {
  for (TaskEdge* edge : in_edges()) { ConsumeRegst("in", edge->GetSoleRegst()); }
}

void AccuracyCompTaskNode::BuildExecGphAndRegst() {
  const auto& op_vec = logical_node()->op_vec();
  CHECK_EQ(op_vec.size(), 1);
  std::shared_ptr<const Operator> accuracy_op = op_vec[0];
  ExecNode* accuracy_node = mut_exec_gph().NewNode();
  accuracy_node->mut_op() = accuracy_op;
  for (const std::string& ibn : accuracy_op->input_bns()) {
    accuracy_node->BindBnWithOneOfTheRegsts(ibn, GetConsumedRegst("in"));
  }
  std::shared_ptr<RegstDesc> accuracy_regst = GetProducedRegst("accuracy");
  CHECK(accuracy_op->pb_output_bns().empty());
  for (const std::string& obn : accuracy_op->output_bns()) {
    accuracy_regst->AddLbi(accuracy_op->BnInOp2Lbi(obn));
    accuracy_node->BindBnWithRegst(obn, accuracy_regst);
  }
  accuracy_node->InferBlobDescs(parallel_ctx());
}

}  // namespace oneflow
