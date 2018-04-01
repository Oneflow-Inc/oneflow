#include "oneflow/core/graph/decode_compute_task_node.h"
#include "oneflow/core/graph/chain_node.h"

namespace oneflow {

void DecodeCompTaskNode::ProduceAllRegstsAndBindEdges() {
  ProduceRegst("data_tmp", 1, 1);
  std::shared_ptr<RegstDesc> out_regst = ProduceRegst("out");
  for (TaskEdge* edge : out_edges()) { edge->AddRegst("out", out_regst); }
}

void DecodeCompTaskNode::ConsumeAllRegsts() {
  if (in_edges().size() == 1) {
    ConsumeRegst("record", SoleInEdge()->GetSoleRegst());
  } else {
    CHECK_EQ(in_edges().size(), 0);
  }
}

void DecodeCompTaskNode::BuildExecGphAndRegst() {
  std::shared_ptr<RegstDesc> out_regst = GetProducedRegst("out");
  std::shared_ptr<RegstDesc> data_tmp_regst = GetProducedRegst("data_tmp");
  ExecNode* node = mut_exec_gph().NewNode();
  node->mut_op() = chain_node()->SoleOp();
  const auto& data_output_lbns = chain_node()->data_output_lbns();
  for (const std::string& obn : node->op()->output_bns()) {
    const std::string& lbn = node->op()->Lbn4BnInOp(obn);
    if (data_output_lbns.find(lbn) == data_output_lbns.end()) {
      data_tmp_regst->AddLbn(lbn);
      node->BindBnInOpAndRegst(obn, data_tmp_regst);
    } else {
      out_regst->AddLbn(lbn);
      node->BindBnInOpAndRegst(obn, out_regst);
    }
  }
  for (const std::string& dtbn : node->op()->data_tmp_bns()) {
    const std::string& lbn = node->op()->Lbn4BnInOp(dtbn);
    data_tmp_regst->AddLbn(lbn);
    node->BindBnInOpAndRegst(dtbn, data_tmp_regst);
  }
  node->InferBlobDescs(parallel_ctx(), device_type());
}

}  // namespace oneflow
