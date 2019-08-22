#include "oneflow/core/graph/logical_node.h"
#include "oneflow/core/graph/sink_compute_task_node.h"

namespace oneflow {

void SinkCompTaskNode::ProduceAllRegstsAndBindEdges() {}

void SinkCompTaskNode::ConsumeAllRegsts() {
  ForEachInDataEdge([&](TaskEdge* edge) { ConsumeRegst("in", edge->GetSoleRegst()); });
}

void SinkCompTaskNode::BuildExecGphAndRegst() {
  ExecNode* node = mut_exec_gph().NewNode();
  node->mut_op() = logical_node()->SoleOp();
  for (const std::string& ibn : node->op()->input_bns()) {
    node->BindBnWithOneOfTheRegsts(ibn, GetConsumedRegst("in"));
  }
  CHECK(node->op()->tmp_bns().empty());
  CHECK(node->op()->output_bns().empty());
}

}  // namespace oneflow
