#include "oneflow/core/graph/decode_compute_task_node.h"
#include "oneflow/core/graph/task_graph.h"
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

void DecodeCompTaskNode::ProduceAllRegstsAndBindEdges() {
  ProduceRegst("data_tmp", 1, 1);
  ProduceB121Regst("out");
  for (TaskEdge* edge : out_edges()) { BindEdgeWithProducedB121Regst(edge, "out"); }
}

void DecodeCompTaskNode::ConsumeAllRegsts() {
  if (in_edges().size() == 1) {
    ConsumeRegst("record", SoleInEdge()->GetSoleRegst());
  } else {
    CHECK_EQ(in_edges().size(), 0);
  }
}

void DecodeCompTaskNode::BuildExecGphAndRegst() {
  std::shared_ptr<RegstDesc> data_tmp_regst = GetProducedRegst("data_tmp");
  ExecNode* node = mut_exec_gph().NewNode();
  node->mut_op() = logical_node()->SoleOp();
  for (const std::string& obn : node->op()->output_bns()) {
    const LogicalBlobId& lbi = node->op()->BnInOp2Lbi(obn);
    if (TryAddLbiToB121RegstAndBindIt(node, obn, "out") == false) {
      data_tmp_regst->AddLbi(lbi);
      node->BindBnWithRegst(obn, data_tmp_regst);
    }
  }
  node->AddBnToRegstAndBindIt(&Operator::data_tmp_bns, data_tmp_regst);
  node->InferBlobDescs(parallel_ctx());
}

}  // namespace oneflow
