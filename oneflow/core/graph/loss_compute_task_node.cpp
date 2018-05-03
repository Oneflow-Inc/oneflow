#include "oneflow/core/graph/loss_compute_task_node.h"
#include "oneflow/core/graph/task_graph.h"
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

void LossCompTaskNode::ProduceAllRegstsAndBindEdges() {
  ProduceRegst("loss");
  ProduceB121Regst("out");
  ProduceRegst("data_tmp", 1, 1);
  for (TaskEdge* edge : out_edges()) {
    const LogicalNode* succ_logical = GetOneSuccLogicalNodeOnEdge(edge);
    if (succ_logical->TypeName() == "LossAcc") {
      BindEdgeWithProducedRegst(edge, "loss");
    } else {
      BindEdgeWithProducedB121Regst(edge, "out");
    }
  }
}

void LossCompTaskNode::ConsumeAllRegsts() {
  for (TaskEdge* edge : in_edges()) { ConsumeRegst("in", edge->GetSoleRegst()); }
}

void LossCompTaskNode::BuildExecGphAndRegst() {
  const auto& op_vec = logical_node()->op_vec();
  if (Global<JobDesc>::Get()->IsTrain()) {
    CHECK_EQ(op_vec.size(), 2);
  } else {
    CHECK_EQ(op_vec.size(), 1);
  }
  std::shared_ptr<const Operator> loss_op = op_vec[0];
  ExecNode* loss_node = mut_exec_gph().NewNode();
  loss_node->mut_op() = loss_op;
  for (const std::string& ibn : loss_op->input_bns()) {
    loss_node->BindBnWithOneOfTheRegsts(ibn, GetConsumedRegst("in"));
  }
  std::shared_ptr<RegstDesc> data_tmp_regst = GetProducedRegst("data_tmp");
  loss_node->AddBnToRegstAndBindIt(&Operator::data_tmp_bns, data_tmp_regst);
  for (const std::string& obn : loss_op->output_bns()) {
    if (!TryAddLbiToB121RegstAndBindIt(loss_node, obn, "out")) {
      data_tmp_regst->AddLbi(loss_op->BnInOp2Lbi(obn));
      loss_node->BindBnWithRegst(obn, data_tmp_regst);
    }
  }
  loss_node->InferBlobDescs(parallel_ctx());

  if (Global<JobDesc>::Get()->IsTrain()) { BuildRegstWhenTraining(); }
}

void LossCompTaskNode::BuildRegstWhenTraining() {
  const auto& op_vec = logical_node()->op_vec();
  ExecNode* loss_node = mut_exec_gph().SoleNode();
  std::shared_ptr<const Operator> loss_op = op_vec[0];
  for (const std::string& idbn : loss_op->input_diff_bns()) {
    TryAddLbiToB121RegstAndBindIt(loss_node, idbn, "out");
  }
  std::shared_ptr<RegstDesc> out_regst_boxing = GetProducedRegst("boxing_out");
  std::shared_ptr<RegstDesc> out_regst_121 = GetProducedRegst("121_out");
  for (std::weak_ptr<RegstDesc> regst : GetConsumedRegst("in")) {
    out_regst_boxing->CopyBlobDescWithoutAddLbi(regst.lock().get());
    out_regst_121->CopyBlobDescWithoutAddLbi(regst.lock().get());
  }

  std::shared_ptr<const Operator> sum_op = op_vec[1];
  ExecNode* sum_node = mut_exec_gph().NewNode();
  sum_node->mut_op() = sum_op;
  Connect(loss_node, mut_exec_gph().NewEdge(), sum_node);

  const LogicalBlobId& sum_ilbi = sum_op->BnInOp2Lbi(sum_op->SoleIbn());
  if (out_regst_boxing->GetBlobDesc(sum_ilbi)) {
    sum_node->BindBnWithRegst(sum_op->SoleIbn(), out_regst_boxing);
  } else if (out_regst_121->GetBlobDesc(sum_ilbi)) {
    sum_node->BindBnWithRegst(sum_op->SoleIbn(), out_regst_121);
  } else {
    sum_node->BindBnWithRegst(sum_op->SoleIbn(), GetProducedRegst("data_tmp"));
  }

  std::shared_ptr<RegstDesc> loss_regst = GetProducedRegst("loss");
  loss_regst->AddLbi(sum_op->BnInOp2Lbi(sum_op->SoleObn()));
  sum_node->BindBnWithRegst(sum_op->SoleObn(), loss_regst);
  if (!loss_op->GetValFromCustomizedConf<std::string>("weight").empty()) {
    loss_regst->AddLbi(loss_op->BnInOp2Lbi("reduction_coefficient"));
    loss_node->BindBnWithRegst("reduction_coefficient", loss_regst);
  }
  sum_node->InferBlobDescs(parallel_ctx());
}

}  // namespace oneflow
