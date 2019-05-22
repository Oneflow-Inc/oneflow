#include "oneflow/core/graph/boxing_v2_task_node.h"

namespace oneflow {

void BoxingV2TaskNode::Init(const LogicalBlobId& lbi, const TensorPartialView& out_view,
                            const BoxingV2TaskMode mode) {
  lbi_ = lbi;
  out_view_ = out_view;
  mode_ = mode;
}

void BoxingV2TaskNode::ProduceAllRegstsAndBindEdges() {
  std::shared_ptr<RegstDesc> out_regst_desc = ProduceRegst("out", false, 1, 1);
  this->ForEachOutDataEdge([&](TaskEdge* edge) { edge->AddRegst("out", out_regst_desc); });
  ProduceRegst("fw_buf", false, 1, 1);
}

void BoxingV2TaskNode::ConsumeAllRegsts() {
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

void BoxingV2TaskNode::BuildExecGphAndRegst() {
  ExecNode* node = mut_exec_gph().NewNode();
  std::shared_ptr<Operator> op = ConstructOp(GetBoxingOpConf());
  node->mut_op() = op;
  FOR_RANGE(size_t, i, 0, op->input_bns().size()) {
    const std::string& ibn = op->input_bns().Get(i);
    CHECK_EQ(GenUnRepeatedBn(ibn).second, i);
    node->BindBnWithRegst(ibn, GetSoleConsumedRegst("in_" + std::to_string(i)));
  }
  std::shared_ptr<RegstDesc> out_regst = GetProducedRegst("out");
  out_regst->AddLbi(lbi_);
  node->BindBnWithRegst(op->SoleObn(), out_regst);
  node->AddBnToRegstAndBindIt(&Operator::fw_buf_bns, GetProducedRegst("fw_buf"));
  node->InferBlobDescs(parallel_ctx());
}

void BoxingV2TaskNode::InferProducedDataRegstTimeShape() { NaiveInferProducedDataRegstTimeShape(); }

void BoxingV2TaskNode::SetInDataEdgeView(const TaskEdge* edge, const TensorPartialView& view) {
  CHECK(in_data_edge2view_.emplace(edge, view).second);
  ordered_in_data_edges_.push_back(edge);
}

OperatorConf BoxingV2TaskNode::GetBoxingOpConf() {
  OperatorConf op_conf{};
  op_conf.set_device_type(device_type());
  BoxingV2Conf boxing_conf{};
  *boxing_conf.mutable_lbi() = lbi_;
  out_view_.ToProto(boxing_conf.mutable_out_view());
  for (const TaskEdge* edge : ordered_in_data_edges_) {
    in_data_edge2view_.at(edge).ToProto(boxing_conf.mutable_in_view()->Add());
  }
  if (mode_ == kBoxingV2TaskModeCopy) {
    op_conf.set_name("System-Boxing-BoxingCopy-" + NewUniqueId());
    BoxingV2CopyOpConf* conf = op_conf.mutable_boxing_v2_copy_conf();
    *conf->mutable_boxing_conf() = boxing_conf;
  } else if (mode_ == kBoxingV2TaskModeAdd) {
    op_conf.set_name("System-Boxing-BoxingAdd-" + NewUniqueId());
    BoxingV2AddOpConf* conf = op_conf.mutable_boxing_v2_add_conf();
    *conf->mutable_boxing_conf() = boxing_conf;
  } else {
    UNIMPLEMENTED();
  }
  return op_conf;
}

}  // namespace oneflow
