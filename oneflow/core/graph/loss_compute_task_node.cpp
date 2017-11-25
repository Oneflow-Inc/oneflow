#include "oneflow/core/graph/loss_compute_task_node.h"
#include "oneflow/core/graph/chain_node.h"

namespace oneflow {

void LossCompTaskNode::ProduceAllRegstsAndBindEdges() {
  auto loss_regst = ProduceRegst("loss", 1, kMaxRegisterNum);
  auto in_diff_regst = ProduceRegst("in_diff", 1, kMaxRegisterNum);
  auto data_tmp_regst = ProduceRegst("data_tmp", 1, kMaxRegisterNum);
  for (TaskEdge* edge : out_edges()) {
    TaskType dst_task_node_type = edge->dst_node()->GetTaskType();
    if (dst_task_node_type == TaskType::kLossAcc) {
      edge->AddRegst("loss", loss_regst);
    } else {
      edge->AddRegst("in_diff", in_diff_regst);
    }
  }
}

void LossCompTaskNode::ConsumeAllRegsts() {
  ConsumeRegst("in", SoleInEdge()->GetSoleRegst());
}

void LossCompTaskNode::BuildExecGphAndRegst() {
  std::shared_ptr<RegstDesc> in_regst = GetConsumedRegst("in");
  std::shared_ptr<RegstDesc> loss_regst = GetProducedRegst("loss");
  std::shared_ptr<RegstDesc> in_diff_regst = GetProducedRegst("in_diff");
  std::shared_ptr<RegstDesc> data_tmp_regst = GetProducedRegst("data_tmp");
  in_diff_regst->CopyBlobDescFrom(in_regst.get());
  std::shared_ptr<const Operator> op = chain_node()->SoleOp();
  CHECK(op->IsLossOp());
  ExecNode* node = mut_exec_gph().NewNode();
  node->mut_op() = op;
  for (const std::string& ibn : op->input_bns()) {
    node->BindBnInOpAndRegst(ibn, in_regst);
  }
  loss_regst->AddLbn(op->Lbn4BnInOp(op->SoleObn()));
  node->BindBnInOpAndRegst(op->SoleObn(), loss_regst);
  for (const std::string& idbn : op->input_diff_bns()) {
    in_diff_regst->AddLbn(op->Lbn4BnInOp(idbn));
    node->BindBnInOpAndRegst(idbn, in_diff_regst);
  }
  for (const std::string& dtbn : op->data_tmp_bns()) {
    data_tmp_regst->AddLbn(op->Lbn4BnInOp(dtbn));
    node->BindBnInOpAndRegst(dtbn, data_tmp_regst);
  }
  op->InferBlobDescs(node->GetBlobDesc4BnInOpFunc(), parallel_ctx());
}

}  // namespace oneflow
