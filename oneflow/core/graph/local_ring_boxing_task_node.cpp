#include <utility>

#include "oneflow/core/graph/local_ring_boxing_task_node.h"

namespace oneflow {

void LocalRingBoxingTaskNode::Init(int64_t machine_id, int64_t thrd_id, const LogicalBlobId& lbi,
                                   LocalRingBoxingTaskNode* lhs, std::vector<int64_t> ring) {
  lbi_ = lbi;
  lhs_task_node_ = lhs;
  ring_ = std::move(ring);
  set_machine_id(machine_id);
  set_thrd_id(thrd_id);
  set_area_id(kMdUpdtArea);
}

bool LocalRingBoxingTaskNode::IsReadyForBuild() { return GetSoleConsumedRegst("in")->IsLocked(); }

void LocalRingBoxingTaskNode::ProduceAllRegstsAndBindEdges() {
  std::shared_ptr<RegstDesc> out_regst_desc = ProduceRegst("out", false, 1, 1);
  this->ForEachOutDataEdge([&](TaskEdge* edge) { edge->AddRegst("out", out_regst_desc); });
  send_regst_desc_ = ProduceRegst("send", false, 1, 1);
}

void LocalRingBoxingTaskNode::ConsumeAllRegsts() {
  ConsumeRegst("in", SoleInDataEdge()->GetSoleRegst());
  ConsumeRegst("recv", lhs_task_node_->send_regst_desc());
}

void LocalRingBoxingTaskNode::BuildExecGphAndRegst() {
  ExecNode* node = mut_exec_gph().NewNode();
  OperatorConf op_conf{};
  op_conf.set_name("System-Boxing-LocalRingBoxing-" + NewUniqueId());
  op_conf.set_device_type(device_type());
  *op_conf.mutable_local_ring_all_reduce_conf()->mutable_local_ring_boxing_conf()->mutable_lbi() =
      lbi_;
  FOR_RANGE(int64_t, i, 0, ring_.size()) {
    *op_conf.mutable_local_ring_all_reduce_conf()
         ->mutable_local_ring_boxing_conf()
         ->mutable_ring()
         ->Add() = ring_.at(i);
  }
  std::shared_ptr<Operator> op = ConstructOp(op_conf);
  node->mut_op() = op;
  node->BindBnWithRegst("in", GetSoleConsumedRegst("in"));
  node->BindBnWithRegst("recv", GetSoleConsumedRegst("recv"));
  std::shared_ptr<RegstDesc> out_regst = GetProducedRegst("out");
  out_regst->AddLbi(lbi_);
  node->BindBnWithRegst("out", out_regst);
  std::shared_ptr<RegstDesc> send_regst = GetProducedRegst("send");
  send_regst->AddLbi(lbi_);
  node->BindBnWithRegst("send", send_regst);
  node->InferBlobDescs(parallel_ctx());
}

void LocalRingBoxingTaskNode::InferProducedDataRegstTimeShape() {
  *GetProducedRegst("out")->mut_data_regst_time_shape() =
      GetSoleConsumedRegst("in")->data_regst_time_shape();
  *GetProducedRegst("send")->mut_data_regst_time_shape() =
      GetSoleConsumedRegst("in")->data_regst_time_shape();
}

}  // namespace oneflow
