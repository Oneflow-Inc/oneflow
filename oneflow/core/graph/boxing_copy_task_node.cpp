#include "oneflow/core/graph/boxing_copy_task_node.h"

namespace oneflow {

void BoxingCopyTaskNode::ProduceAllRegstsAndBindEdges() {
  std::shared_ptr<RegstDesc> out_regst_desc = ProduceRegst("out", false, 1, 1);
  this->ForEachOutDataEdge([&](TaskEdge* edge) { edge->AddRegst("out", out_regst_desc); });
  ProduceRegst("fw_buf", false, 1, 1);
}

void BoxingCopyTaskNode::ConsumeAllRegsts() {
  HashMap<const TaskEdge*, int64_t> edge2order_;
  FOR_RANGE(int64_t, i, 0, sorted_in_data_edge_vec_.size()) {
    edge2order_.emplace(sorted_in_data_edge_vec_.at(i), i);
  }
  int64_t in_data_edge_cnt = 0;
  ForEachInDataEdge([&](TaskEdge* edge) {
    const auto order_it = edge2order_.find(edge);
    CHECK(order_it != edge2order_.end());
    ConsumeRegst("in_" + std::to_string(order_it->second), edge->GetSoleRegst());
    in_data_edge_cnt += 1;
  });
  CHECK_EQ(in_data_edge_cnt, sorted_in_data_edge_vec_.size());
}

void BoxingCopyTaskNode::BuildExecGphAndRegst() {
  OperatorConf op_conf{};
  op_conf.set_name("System-Boxing-Copy-" + NewUniqueId());
  BoxingCopyOpConf* boxing_copy_conf = op_conf.mutable_boxing_copy_conf();
  *boxing_copy_conf->mutable_lbi() = lbi_;
  out_view_.ToProto(boxing_copy_conf->mutable_out_view());
  for (const TaskEdge* edge : sorted_in_data_edge_vec_) {
    in_data_edge2tensor_partial_view_.at(edge).ToProto(boxing_copy_conf->mutable_in_view()->Add());
  }
  ExecNode* node = mut_exec_gph().NewNode();
  std::shared_ptr<Operator> op = ConstructOp(op_conf);
  node->mut_op() = op;
  FOR_RANGE(size_t, i, 0, op->input_bns().size()) {
    const std::string& ibn = op->input_bns().Get(i);
    CHECK_EQ(GenUnRepeatedBn(ibn).second, i);
    node->BindBnWithRegst(ibn, GetSoleConsumedRegst("in_" + std::to_string(i)));
  }
  std::shared_ptr<RegstDesc> out_regst = GetProducedRegst("out");
  out_regst->AddLbi(op->BnInOp2Lbi(op->SoleObn()));
  node->BindBnWithRegst(op->SoleObn(), out_regst);
  node->AddBnToRegstAndBindIt(&Operator::fw_buf_bns, GetProducedRegst("fw_buf"));
  node->InferBlobDescs(parallel_ctx());
}

void BoxingCopyTaskNode::InferProducedDataRegstTimeShape() {
  NaiveInferProducedDataRegstTimeShape();
}

void BoxingCopyTaskNode::BindTensorPartialViewToInDataEdge(const TaskEdge* edge,
                                                           const TensorPartialView& view) {
  sorted_in_data_edge_vec_.push_back(edge);
  in_data_edge2tensor_partial_view_.emplace(edge, view);
}

void BoxingCopyTaskNode::SetOutTensorPartialView(const TensorPartialView& out_view) {
  out_view_ = out_view;
}

void BoxingCopyTaskNode::SetLbi(const LogicalBlobId& lbi) { lbi_ = lbi; }

}  // namespace oneflow
