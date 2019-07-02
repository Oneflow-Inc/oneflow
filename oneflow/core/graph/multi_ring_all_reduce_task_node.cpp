#include "oneflow/core/graph/multi_ring_all_reduce_task_node.h"

namespace oneflow {

void MultiRingAllReduceTaskNode::Init(int64_t machine_id, int64_t thrd_id, const LogicalBlobId& lbi,
                                      const Shape& logical_blob_shape,
                                      const ParallelContext& parallel_ctx) {
  lbi_ = lbi;
  set_machine_id(machine_id);
  set_thrd_id(thrd_id);
  set_area_id(kMdUpdtArea);
  parallel_ctx_ = parallel_ctx;
  logical_blob_shape_ = logical_blob_shape;
}

bool MultiRingAllReduceTaskNode::IsReadyForBuild() {
  return GetSoleConsumedRegst("in")->IsLocked();
}

void MultiRingAllReduceTaskNode::ProduceAllRegstsAndBindEdges() {
  std::shared_ptr<RegstDesc> out_regst_desc = ProduceRegst("out", false, 1, 1);
  HashMap<TaskNode*, std::shared_ptr<RegstDesc>> send_to_node2regst;
  FOR_RANGE(int64_t, i, 0, send_to_.size()) {
    std::shared_ptr<RegstDesc> send_regst_desc =
        ProduceRegst("send_" + std::to_string(i), false, 1, 1);
    CHECK(send_to_node2regst.emplace(send_to_.at(i), send_regst_desc).second);
  }
  this->ForEachOutDataEdge([&](TaskEdge* edge) {
    auto it = send_to_node2regst.find(edge->dst_node());
    if (it == send_to_node2regst.cend()) {
      edge->AddRegst("out", out_regst_desc);
    } else {
      edge->AddRegst("send", it->second);
    }
  });
}

void MultiRingAllReduceTaskNode::ConsumeAllRegsts() {
  FOR_RANGE(int64_t, i, 0, recv_from_.size()) {
    ConsumeRegst("recv_" + std::to_string(i), recv_from_.at(i)->GetProducedRegst("copy_out"));
  }
  ConsumeRegst("in", SoleInDataEdge()->GetSoleRegst());
}

void MultiRingAllReduceTaskNode::AddRing(const std::vector<int64_t>& ring_next, TaskNode* send_to,
                                         TaskNode* recv_from) {
  rings_.push_back(ring_next);
  send_to_.push_back(send_to);
  recv_from_.push_back(recv_from);
}

void MultiRingAllReduceTaskNode::BuildExecGphAndRegst() {
  ExecNode* node = mut_exec_gph().NewNode();
  std::shared_ptr<Operator> op = ConstructOp(GenOpConf());
  node->mut_op() = op;
  node->BindBnWithRegst("in", GetSoleConsumedRegst("in"));
  std::shared_ptr<RegstDesc> out_regst = GetProducedRegst("out");
  out_regst->AddLbi(lbi_);
  node->BindBnWithRegst("out", out_regst);
  FOR_RANGE(int64_t, i, 0, rings_.size()) {
    const std::string recv_name = "recv_" + std::to_string(i);
    const std::string send_name = "send_" + std::to_string(i);
    node->BindBnWithRegst(recv_name, GetSoleConsumedRegst(recv_name));
    std::shared_ptr<RegstDesc> send_regst = GetProducedRegst(send_name);
    send_regst->AddLbi(lbi_);
    node->BindBnWithRegst(send_name, send_regst);
  }

  node->InferBlobDescs(parallel_ctx());
}

void MultiRingAllReduceTaskNode::InferProducedDataRegstTimeShape() {
  *GetProducedRegst("out")->mut_data_regst_time_shape() =
      GetSoleConsumedRegst("in")->data_regst_time_shape();
  FOR_RANGE(int64_t, i, 0, send_to_.size()) {
    *GetProducedRegst("send_" + std::to_string(i))->mut_data_regst_time_shape() =
        GetSoleConsumedRegst("in")->data_regst_time_shape();
  }
}

OperatorConf MultiRingAllReduceTaskNode::GenOpConf() const {
  OperatorConf op_conf{};
  op_conf.set_name("System-Boxing-MultiRingAllReduce-" + NewUniqueId());
  op_conf.set_device_type(device_type());
  MultiRingAllReduceOpConf* all_reduce_conf = op_conf.mutable_multi_ring_all_reduce_conf();
  *all_reduce_conf->mutable_lbi() = lbi_;
  FOR_RANGE(int64_t, ring_id, 0, rings_.size()) {
    RingConf ring_conf{};
    for (const int64_t id : rings_.at(ring_id)) { ring_conf.mutable_next()->Add(id); }
    *all_reduce_conf->mutable_rings()->Add() = ring_conf;
  }
  return op_conf;
}

void MultiRingAllReduceTaskNode::ToProto(TaskProto* task_proto) {
  TaskNode::ToProto(task_proto);
  *(task_proto->mutable_parallel_ctx()) = parallel_ctx_;
}

}  // namespace oneflow
