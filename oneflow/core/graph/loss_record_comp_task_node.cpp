#include "oneflow/core/graph/loss_record_comp_task_node.h"
#include "oneflow/core/graph/loss_record_task_graph.h"

namespace oneflow {

void LossRecordCompTaskNode::BuildExecAndEnrollLbn2Regsts(TaskGraph* gph) {
  if (chain_node()->op_vec().empty()) {
    auto loss_record_gph = static_cast<LossRecordTaskGraph*>(gph);
    CompTaskNode* loss_acc_task =
        loss_record_gph->GetLossAccCompTaskNodeFromParallelId(parallel_id());
    auto loss_acc_regst = loss_acc_task->GetProducedRegstDesc("loss_acc");
    BindProducedRegstAndOutEdge(loss_acc_regst, SoleOutEdge());
    return;
  }
  auto loss_acc_regst = GetRelatedRegst(SoleInEdge());
  ExecNode* exec_node = mut_exec_gph().NewNode();
  exec_node->mut_op() = chain_node()->SoleOp();
  exec_node->BindBnInOpAndRegst(exec_node->op()->SoleIbn(), loss_acc_regst);
  ConsumeRegstDesc("loss_acc", loss_acc_regst);
  mut_exec_gph().UpdateSourceAndSink();
}

void LossRecordCompTaskNode::InferShapeOfBlobsInProducedRegsts(TaskGraph* gph) {
}

}  // namespace oneflow
