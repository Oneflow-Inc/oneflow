#include "oneflow/core/graph/loss_compute_task_node.h"
#include "oneflow/core/graph/task_graph.h"
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

void LossCompTaskNode::ProduceAllRegstsAndBindEdges() {
  ProduceRegst("loss", false);
  ProduceRegst("out", true);
  ProduceRegst("data_tmp", true, 1, 1);
  for (TaskEdge* edge : out_edges()) {
    const LogicalNode* succ_logical = GetOneSuccLogicalNodeOnEdge(edge);
    if (succ_logical->TypeName() == "LossAcc") {
      BindEdgeWithProducedRegst(edge, "loss");
    } else {
      BindEdgeWithProducedRegst(edge, "out");
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

  if (Global<JobDesc>::Get()->IsTrain()) {
    BuildRegstWhenTraining();
  } else {
    std::shared_ptr<RegstDesc> out_regst = GetProducedRegst("out");
    for (const std::string& obn : loss_op->output_bns()) {
      out_regst->AddLbi(loss_op->BnInOp2Lbi(obn));
      loss_node->BindBnWithRegst(obn, out_regst);
    }
  }
  mut_exec_gph().TopoForEachNode([this](ExecNode* node) { node->InferBlobDescs(parallel_ctx()); });
  FOR_EACH(input_diff_bn_it, loss_op->input_diff_bns()) {
    GetProducedRegst("out")
        ->MutBlobDesc(loss_op->BnInOp2Lbi(*input_diff_bn_it))
        ->set_has_loss_instance_num_field(true);
  }
}

void LossCompTaskNode::BuildRegstWhenTraining() {
  const auto& op_vec = logical_node()->op_vec();
  ExecNode* loss_node = mut_exec_gph().SoleNode();
  std::shared_ptr<const Operator> loss_op = op_vec[0];
  std::shared_ptr<RegstDesc> out_regst = GetProducedRegst("out");
  std::shared_ptr<RegstDesc> data_tmp_regst = GetProducedRegst("data_tmp");
  loss_node->AddBnToRegstAndBindIt(&Operator::input_diff_bns, out_regst);
  for (std::shared_ptr<RegstDesc> regst : GetConsumedRegst("in")) {
    out_regst->CopyBlobDescWithoutAddLbi(regst.get());
  }
  data_tmp_regst->AddLbi(loss_op->BnInOp2Lbi("loss"));
  loss_node->BindBnWithRegst("loss", data_tmp_regst);

  std::shared_ptr<const Operator> sum_op = op_vec[1];
  ExecNode* sum_node = mut_exec_gph().NewNode();
  sum_node->mut_op() = sum_op;
  Connect(loss_node, mut_exec_gph().NewEdge(), sum_node);

  sum_node->BindBnWithRegst(sum_op->SoleIbn(), data_tmp_regst);
  sum_node->AddBnToRegstAndBindIt(&Operator::data_tmp_bns, data_tmp_regst);

  std::shared_ptr<RegstDesc> loss_regst = GetProducedRegst("loss");
  loss_regst->AddLbi(sum_op->BnInOp2Lbi(sum_op->SoleObn()));
  loss_regst->AddLbi(loss_op->BnInOp2Lbi("loss_instance_num"));
  sum_node->BindBnWithRegst(sum_op->SoleObn(), loss_regst);
  loss_node->BindBnWithRegst("loss_instance_num", loss_regst);
  if (!loss_op->GetValFromCustomizedConf<std::string>("weight").empty()) {
    loss_regst->AddLbi(loss_op->BnInOp2Lbi("reduction_coefficient"));
    loss_node->BindBnWithRegst("reduction_coefficient", loss_regst);
  }
}

void LossCompTaskNode::InferProducedDataRegstTimeShape() { NaiveInferProducedDataRegstTimeShape(); }

}  // namespace oneflow
