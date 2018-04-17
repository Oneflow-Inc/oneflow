#include "oneflow/core/graph/loss_compute_task_node.h"
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

void LossCompTaskNode::ProduceAllRegstsAndBindEdges() {
  auto loss_regst = ProduceRegst("loss");
  auto in_diff_regst = ProduceRegst("in_diff");
  auto data_tmp_regst = ProduceRegst("data_tmp", 1, 1);
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
  // regst
  std::shared_ptr<RegstDesc> in_regst = GetSoleConsumedRegst("in");
  std::shared_ptr<RegstDesc> loss_regst = GetProducedRegst("loss");
  std::shared_ptr<RegstDesc> in_diff_regst = GetProducedRegst("in_diff");
  std::shared_ptr<RegstDesc> data_tmp_regst = GetProducedRegst("data_tmp");
  // op
  const auto& op_vec = logical_node()->op_vec();
  CHECK_EQ(op_vec.size(), 2);
  std::shared_ptr<const Operator> loss_op = op_vec[0];
  std::shared_ptr<const Operator> sum_op = op_vec[1];
  // exec gph
  ExecNode* loss_node = mut_exec_gph().NewNode();
  loss_node->mut_op() = loss_op;
  ExecNode* sum_node = mut_exec_gph().NewNode();
  sum_node->mut_op() = sum_op;
  Connect(loss_node, mut_exec_gph().NewEdge(), sum_node);
  // bind
  for (const std::string& ibn : loss_op->input_bns()) {
    loss_node->BindBnInOpAndRegst(ibn, in_regst);
  }
  for (const std::string& obn : loss_op->output_bns()) {
    data_tmp_regst->AddLbi(loss_op->BnInOp2Lbi(obn));
    loss_node->BindBnInOpAndRegst(obn, data_tmp_regst);
  }
  for (const std::string& idbn : loss_op->input_diff_bns()) {
    in_diff_regst->AddLbi(loss_op->BnInOp2Lbi(idbn));
    loss_node->BindBnInOpAndRegst(idbn, in_diff_regst);
  }
  for (const std::string& dtbn : loss_op->data_tmp_bns()) {
    data_tmp_regst->AddLbi(loss_op->BnInOp2Lbi(dtbn));
    loss_node->BindBnInOpAndRegst(dtbn, data_tmp_regst);
  }
  sum_node->BindBnInOpAndRegst(sum_op->SoleIbn(), data_tmp_regst);
  loss_regst->AddLbi(sum_op->BnInOp2Lbi(sum_op->SoleObn()));
  sum_node->BindBnInOpAndRegst(sum_op->SoleObn(), loss_regst);
  if (!loss_op->GetValFromCustomizedConf<std::string>("weight").empty()) {
    loss_regst->AddLbi(loss_op->BnInOp2Lbi("reduction_coefficient"));
    loss_node->BindBnInOpAndRegst("reduction_coefficient", loss_regst);
  }
  loss_node->InferBlobDescs(parallel_ctx());
  sum_node->InferBlobDescs(parallel_ctx());
  in_diff_regst->CopyBlobDescWithoutAddLbi(in_regst.get());
}

}  // namespace oneflow
