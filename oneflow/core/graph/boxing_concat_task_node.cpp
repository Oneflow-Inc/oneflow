#include "oneflow/core/graph/boxing_concat_task_node.h"

namespace oneflow {

void BoxingConcatTaskNode::Init(const LogicalBlobId& lbi, int64_t machine_id, int64_t thrd_id,
                                int64_t axis) {
  lbi_ = lbi;
  axis_ = axis;
  set_machine_id(machine_id);
  set_thrd_id(thrd_id);
  set_area_id(kDataForwardArea);
}

void BoxingConcatTaskNode::ConnectToSrc(TaskNode* src, TaskEdge* edge) {
  Connect<TaskNode>(src, edge, this);
  ordered_in_data_edges_.push_back(edge);
}

void BoxingConcatTaskNode::ProduceAllRegstsAndBindEdges() {
  std::shared_ptr<RegstDesc> out_regst_desc = ProduceRegst("out", false, 1, 1);
  this->ForEachOutDataEdge([&](TaskEdge* edge) { edge->AddRegst("out", out_regst_desc); });
}

void BoxingConcatTaskNode::ConsumeAllRegsts() {
  HashMap<const TaskEdge*, int64_t> edge2order_;
  FOR_RANGE(int64_t, i, 0, ordered_in_data_edges_.size()) {
    edge2order_.emplace(ordered_in_data_edges_.at(i), i);
  }
  int64_t in_data_edge_cnt = 0;
  ForEachInDataEdge([&](TaskEdge* edge) {
    const auto order_it = edge2order_.find(edge);
    CHECK(order_it != edge2order_.end());
    ConsumeRegst("in_" + std::to_string(order_it->second), edge->GetSoleRegst());
    in_data_edge_cnt += 1;
  });
  CHECK_EQ(in_data_edge_cnt, ordered_in_data_edges_.size());
}

void BoxingConcatTaskNode::BuildExecGphAndRegst() {
  ExecNode* node = mut_exec_gph().NewNode();
  std::shared_ptr<Operator> op = ConstructOp(GetConcatOpConf());
  node->mut_op() = op;
  FOR_RANGE(size_t, i, 0, op->input_bns().size()) {
    const std::string& ibn = op->input_bns().Get(i);
    CHECK_EQ(GenUnRepeatedBn(ibn).second, i);
    node->BindBnWithRegst(ibn, GetSoleConsumedRegst("in_" + std::to_string(i)));
  }
  std::shared_ptr<RegstDesc> out_regst = GetProducedRegst("out");
  out_regst->AddLbi(lbi_);
  node->BindBnWithRegst(op->SoleObn(), out_regst);
  node->InferBlobDescs(parallel_ctx());
}

void BoxingConcatTaskNode::InferProducedDataRegstTimeShape() {
  NaiveInferProducedDataRegstTimeShape();
}

OperatorConf BoxingConcatTaskNode::GetConcatOpConf() {
  OperatorConf op_conf{};
  op_conf.set_name("System-Boxing-BoxingConcat-" + NewUniqueId());
  op_conf.set_device_type(device_type());
  BoxingConcatOpConf* boxing_concat_conf = op_conf.mutable_boxing_concat_conf();
  boxing_concat_conf->set_axis(axis_);
  *boxing_concat_conf->mutable_lbi() = lbi_;
  boxing_concat_conf->set_in_num(ordered_in_data_edges_.size());
  return op_conf;
}

}  // namespace oneflow
