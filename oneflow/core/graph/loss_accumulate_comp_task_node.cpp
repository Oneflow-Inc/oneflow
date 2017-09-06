#include "oneflow/core/graph/loss_accumulate_comp_task_node.h"
#include "oneflow/core/graph/loss_accumulate_task_graph.h"

namespace oneflow {

void LossAccCompTaskNode::BuildExecAndEnrollLbn2Regsts(TaskGraph* gph) {
  if (chain_node()->op_vec().empty()) {
    CompTaskNode* loss_task = static_cast<LossAccTaskGraph*>(gph)->loss_task();
    auto loss_regst = loss_task->GetProducedRegstDesc("loss");
    BindProducedRegstAndOutEdge(loss_regst, SoleOutEdge());
    return;
  }
  NewProducedRegstDesc("loss_acc", 1);
  auto loss_regst = GetRelatedRegst(SoleInEdge());
  auto loss_acc_regst = GetProducedRegstDesc("loss_acc");
  ExecNode* exec_node = mut_exec_gph().NewNode();
  exec_node->mut_op() = chain_node()->SoleOp();
  exec_node->BindBnInOpAndRegst(exec_node->op()->SoleIbn(), loss_regst);
  exec_node->BindBnInOpAndRegst(exec_node->op()->SoleObn(), loss_acc_regst);
  ConsumeRegstDesc("loss", loss_regst);
  loss_acc_regst->CopyLbnFrom(loss_regst.get());
  mut_exec_gph().UpdateSourceAndSink();
}

void LossAccCompTaskNode::InferBlobDescInProducedRegsts(TaskGraph* gph) {
  if (!chain_node()->op_vec().empty()) {
    auto loss_regst = GetConsumedRegstDesc("loss");
    auto loss_acc_regst = GetProducedRegstDesc("loss_acc");
    loss_acc_regst->CopyBlobDescFrom(loss_regst.get());
  }
}

}  // namespace oneflow
