#include "oneflow/core/graph/boxing_copy_task_node.h"

namespace oneflow {

void BoxingCopyTaskNode::ProduceAllRegstsAndBindEdges() {
  std::shared_ptr<RegstDesc> out_regst_desc = ProduceRegst("out", false, 1, 1);
  this->ForEachOutDataEdge([&](TaskEdge* edge) { edge->AddRegst("out", out_regst_desc); });
  ProduceRegst("fw_buf", false, 1, 1);
}

void BoxingCopyTaskNode::ConsumeAllRegsts() {
  HashMap<const TaskNode*, int64_t> task_node2order_;
  FOR_RANGE(int64_t, i, 0, sorted_pred_task_node_vec_.size()) {
    task_node2order_.emplace(sorted_pred_task_node_vec_.at(i), i);
  }
  int64_t in_data_edge_cnt = 0;
  ForEachInDataEdge([&](TaskEdge* edge) {
    const auto order_it = task_node2order_.find(edge->src_node());
    CHECK(order_it != task_node2order_.end());
    ConsumeRegst("in_" + std::to_string(order_it->second), edge->GetSoleRegst());
    in_data_edge_cnt += 1;
  });
  CHECK_EQ(in_data_edge_cnt, sorted_pred_task_node_vec_.size());
}

void BoxingCopyTaskNode::BuildExecGphAndRegst() {
  ExecNode* node = mut_exec_gph().NewNode();
  node->mut_op() = op_;
  FOR_RANGE(size_t, i, 0, op_->input_bns().size()) {
    const std::string& ibn = op_->input_bns().Get(i);
    CHECK_EQ(GenUnRepeatedBn(ibn).second, i);
    node->BindBnWithRegst(ibn, GetSoleConsumedRegst("in_" + std::to_string(i)));
  }
  std::shared_ptr<RegstDesc> out_regst = GetProducedRegst("out");
  out_regst->AddLbi(op_->BnInOp2Lbi(op_->SoleObn()));
  node->BindBnWithRegst(op_->SoleObn(), out_regst);
  node->AddBnToRegstAndBindIt(&Operator::fw_buf_bns, GetProducedRegst("fw_buf"));
  node->InferBlobDescs(parallel_ctx());
}

void BoxingCopyTaskNode::InferProducedDataRegstTimeShape() {
  NaiveInferProducedDataRegstTimeShape();
}

}  // namespace oneflow
