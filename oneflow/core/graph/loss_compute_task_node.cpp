#include "oneflow/core/graph/loss_compute_task_node.h"
#include "oneflow/core/graph/chain_node.h"

namespace oneflow {

void LossCompTaskNode::ProduceAllRegstsAndBindEdges() {
  auto loss_regst = ProduceRegst("loss", 1, kMaxRegisterNum);
  auto diff_regst = ProduceRegst("in_diff", 1, kMaxRegisterNum);
  auto data_tmp_regst = ProduceRegst("data_tmp", 1, kMaxRegisterNum);
  for (TaskEdge* edge : out_edges()) {
    edge->AddRegst("loss", loss_regst);
    edge->AddRegst("in_diff", diff_regst);
  }
}

void LossCompTaskNode::ConsumeAllRegsts() {
  for (TaskEdge* edge : in_edges()) {
    ConsumeRegst("in", edge->GetSoleRegst());
  }
}

void LossCompTaskNode::BuildExecGphAndRegst() {
  auto loss_regst = GetProducedRegst("loss");
  auto diff_regst = GetProducedRegst("in_diff");
  auto data_tmp_regst = GetProducedRegst("data_tmp");
  ExecNode* loss_node = mut_exec_gph().NewNode();
  loss_node->mut_op() = chain_node()->SoleOp();
  CHECK(loss_node->op()->IsLossOp());
  loss_node->BindBnInOpAndRegst(loss_node->op()->SoleObn(), loss_regst);
  for (const std::string& input_diff_bn : loss_node->op()->input_diff_bns()) {
    loss_node->BindBnInOpAndRegst(input_diff_bn, diff_regst);
  }
  for (const std::string& data_tmp_bn : loss_node->op()->data_tmp_bns()) {
    loss_node->BindBnInOpAndRegst(data_tmp_bn, data_tmp_regst);
  }
  loss_node->op()->InferBlobDescs(loss_node->GetBlobDesc4BnInOpFunc(),
                                  parallel_ctx());
}

}  // namespace oneflow
