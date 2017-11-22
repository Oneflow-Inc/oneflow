#include "oneflow/core/graph/loss_compute_task_node.h"
#include "oneflow/core/graph/chain_node.h"

namespace oneflow {

void LossCompTaskNode::ProduceAllRegstsAndBindEdges() {
  auto loss_regst = ProduceRegst("loss", 1, kMaxRegisterNum);
  auto in_diff_regst = ProduceRegst("in_diff", 1, kMaxRegisterNum);
  auto data_tmp_regst = ProduceRegst("data_tmp", 1, kMaxRegisterNum);
  int dst_loss_node_num = 0;
  for (TaskEdge* edge : out_edges()) {
    // loss_acc_task_node
    if (edge->dst_node()->GetTaskType() == TaskType::kLossAcc) {
      edge->AddRegst("loss", loss_regst);
      dst_loss_node_num++;
    } else {  // boxing_task_node or bw_cmp_task_node
      edge->AddRegst("in_diff", in_diff_regst);
    }
  }
  CHECK_EQ(dst_loss_node_num, 1);
}

void LossCompTaskNode::ConsumeAllRegsts() {
  ConsumeRegst("in", SoleInEdge()->GetSoleRegst());
  /*
    for (TaskEdge* in_edge : in_edges()) {
      // find pred compute task node
      TaskNode* src_node = in_edge->src_node();
      while(dynamic_cast<CompTaskNode*>(src_node) == nullptr) {
        src_node = (*(src_node->in_edges().begin()))->src_node();
      }

      ConsumeRegst("in", edge->GetSoleRegst());
    }
  */
}

void LossCompTaskNode::BuildExecGphAndRegst() {
  auto in_regst = GetConsumedRegst("in");
  auto loss_regst = GetProducedRegst("loss");
  auto in_diff_regst = GetProducedRegst("in_diff");
  auto data_tmp_regst = GetProducedRegst("data_tmp");
  ExecNode* node = mut_exec_gph().NewNode();
  node->mut_op() = chain_node()->SoleOp();
  auto op = node->op();
  CHECK(op->IsLossOp());
  for (const std::string& ibn : op->input_bns()) {
    node->BindBnInOpAndRegst(ibn, in_regst);
  }
  loss_regst->AddLbn(op->Lbn4BnInOp(op->SoleObn()));
  node->BindBnInOpAndRegst(op->SoleObn(), loss_regst);
  for (const std::string& in_diff_bn : op->input_diff_bns()) {
    in_diff_regst->AddLbn(op->Lbn4BnInOp(in_diff_bn));
    node->BindBnInOpAndRegst(in_diff_bn, in_diff_regst);
  }
  for (const std::string& data_tmp_bn : op->data_tmp_bns()) {
    data_tmp_regst->AddLbn(op->Lbn4BnInOp(data_tmp_bn));
    node->BindBnInOpAndRegst(data_tmp_bn, data_tmp_regst);
  }
  op->InferBlobDescs(node->GetBlobDesc4BnInOpFunc(), parallel_ctx());
}

}  // namespace oneflow
