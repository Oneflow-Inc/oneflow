#include "oneflow/core/graph/logical_node.h"
#include "oneflow/core/graph/sink_compute_task_node.h"

namespace oneflow {

void SinkCompTaskNode::ProduceAllRegstsAndBindEdges() {}

void SinkCompTaskNode::ConsumeAllRegsts() {
  for (TaskEdge* edge : in_edges()) {
    edge->ForEachRegst(
        [&](std::shared_ptr<RegstDesc> regst_desc) { ConsumeRegst("in", regst_desc); });
  }
}

void SinkCompTaskNode::BuildExecGphAndRegst() {
  ExecNode* node = mut_exec_gph().NewNode();
  node->mut_op() = logical_node()->SoleOp();
  node->op()->ForEachInputBn(
      [&](const std::string& ibn) { node->BindBnWithOneOfTheRegsts(ibn, GetConsumedRegst("in")); });
  CHECK(node->op()->data_tmp_bns().empty());
  CHECK(node->op()->output_bns().empty());
}

}  // namespace oneflow
