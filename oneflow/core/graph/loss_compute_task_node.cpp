#include "oneflow/core/graph/loss_compute_task_node.h"
#include "oneflow/core/graph/task_graph.h"
#include "oneflow/core/graph/logical_node.h"
#include "oneflow/core/graph/op_graph.h"

namespace oneflow {

void LossCompTaskNode::SetIfComputeHalfLoss() {
  if_comp_half_loss_ =
      (Global<OpGraph>::Get()->GetBlobDataType(logical_node()->op_vec()[0]->BnInOp2Lbi("loss"))
       == DataType::kFloat16);
}

void LossCompTaskNode::ProduceAllRegstsAndBindEdges() {
  SetIfComputeHalfLoss();
  ProduceRegst("loss", false);
  ProduceRegst("out", true);
  ProduceRegst("data_tmp", true, 1, 1);
  ProduceRegst("float_loss", true, 1, 1);
  ProduceRegst("const_buf", false, 1, 1);
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
    if (if_comp_half_loss_) {
      CHECK_EQ(op_vec.size(), 3);
    } else {
      CHECK_EQ(op_vec.size(), 2);
    }
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
  loss_node->AddBnToRegstAndBindIt(&Operator::const_buf_bns, GetProducedRegst("const_buf"));

  if (Global<JobDesc>::Get()->IsTrain()) {
    BuildRegstWhenTraining();
  } else {
    CHECK(!if_comp_half_loss_);
    std::shared_ptr<RegstDesc> out_regst = GetProducedRegst("out");
    for (const std::string& obn : loss_op->output_bns()) {
      out_regst->AddLbi(loss_op->BnInOp2Lbi(obn));
      loss_node->BindBnWithRegst(obn, out_regst);
    }
  }
  mut_exec_gph().TopoForEachNode([this](ExecNode* node) { node->InferBlobDescs(parallel_ctx()); });
  mut_exec_gph().TopoForEachNode(
      [this](ExecNode* node) { node->FixInDiffBlobDescs(parallel_ctx()); });
}

void LossCompTaskNode::BuildRegstWhenTraining() {
  const auto& op_vec = logical_node()->op_vec();
  ExecNode* loss_node = mut_exec_gph().SoleNode();
  std::shared_ptr<const Operator> loss_op = op_vec[0];
  std::shared_ptr<RegstDesc> out_regst = GetProducedRegst("out");
  std::shared_ptr<RegstDesc> data_tmp_regst = GetProducedRegst("data_tmp");
  std::shared_ptr<RegstDesc> float_loss_regst = GetProducedRegst("float_loss");
  std::shared_ptr<RegstDesc> loss_regst = GetProducedRegst("loss");
  loss_node->AddBnToRegstAndBindIt(&Operator::input_diff_bns, out_regst);
  for (std::shared_ptr<RegstDesc> regst : GetConsumedRegst("in")) {
    out_regst->CopyBlobDescWithoutAddLbi(regst.get());
  }
  const LogicalBlobId& loss_lbi = loss_op->BnInOp2Lbi("loss");
  data_tmp_regst->AddLbi(loss_lbi);
  loss_node->BindBnWithRegst("loss", data_tmp_regst);

  std::shared_ptr<Operator> sum_op = nullptr;
  ExecNode* sum_node = mut_exec_gph().NewNode();

  if (if_comp_half_loss_) {
    sum_op = op_vec[2];
    sum_node->mut_op() = sum_op;
    std::shared_ptr<Operator> cast_loss_op = op_vec[1];
    ExecNode* cast_loss_node = mut_exec_gph().NewNode();
    cast_loss_node->mut_op() = cast_loss_op;
    Connect(loss_node, mut_exec_gph().NewEdge(), cast_loss_node);
    cast_loss_node->BindBnWithRegst(cast_loss_op->SoleIbn(), data_tmp_regst);
    cast_loss_node->AddBnToRegstAndBindIt(&Operator::output_bns, float_loss_regst);
    Connect(cast_loss_node, mut_exec_gph().NewEdge(), sum_node);
    sum_node->BindBnWithRegst(sum_op->SoleIbn(), float_loss_regst);
    sum_node->AddBnToRegstAndBindIt(&Operator::data_tmp_bns, float_loss_regst);
  } else {
    sum_op = op_vec[1];
    sum_node->mut_op() = sum_op;
    Connect(loss_node, mut_exec_gph().NewEdge(), sum_node);
    sum_node->BindBnWithRegst(sum_op->SoleIbn(), data_tmp_regst);
    sum_node->AddBnToRegstAndBindIt(&Operator::data_tmp_bns, data_tmp_regst);
  }

  sum_node->AddBnToRegstAndBindIt(&Operator::output_bns, loss_regst);
  loss_node->AddBnToRegstAndBindIt("loss_instance_num", loss_regst);
  if (!loss_op->GetValFromCustomizedConf<std::string>("weight").empty()) {
    CHECK(!if_comp_half_loss_);
    loss_node->AddBnToRegstAndBindIt("reduction_coefficient", loss_regst);
  }
}

void LossCompTaskNode::InferProducedDataRegstTimeShape() { NaiveInferProducedDataRegstTimeShape(); }

}  // namespace oneflow
