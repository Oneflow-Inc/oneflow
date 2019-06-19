#include "oneflow/core/graph/ring_boxing_task_node.h"

namespace oneflow {

void RingBoxingTaskNode::Init(RingBoxingTaskMode mode, int64_t machine_id, int64_t thrd_id,
                              const LogicalBlobId& lbi, const Shape& logical_blob_shape,
                              TaskNode* send_to, TaskNode* recv_from,
                              const std::vector<TensorSliceView>& slices,
                              const std::vector<int64_t>& ring,
                              const ParallelContext& parallel_ctx) {
  mode_ = mode;
  CHECK_EQ(slices.size(), parallel_ctx.parallel_num());
  CHECK_EQ(ring.size(), parallel_ctx.parallel_num());
  HashSet<int64_t> all_parallel_id;
  FOR_RANGE(int64_t, i, 0, parallel_ctx.parallel_num()) {
    CHECK_GE(ring.at(i), 0);
    CHECK_LT(ring.at(i), parallel_ctx.parallel_num());
    CHECK(all_parallel_id.emplace(ring.at(i)).second);
  }
  lbi_ = lbi;
  logical_blob_shape_ = logical_blob_shape;
  slices_ = slices;
  ring_ = ring;
  set_machine_id(machine_id);
  set_thrd_id(thrd_id);
  set_area_id(kMdUpdtArea);
  send_to_ = send_to;
  recv_from_ = recv_from;
  parallel_ctx_ = parallel_ctx;
}

bool RingBoxingTaskNode::IsReadyForBuild() { return GetSoleConsumedRegst("in")->IsLocked(); }

void RingBoxingTaskNode::ProduceAllRegstsAndBindEdges() {
  std::shared_ptr<RegstDesc> out_regst_desc = ProduceRegst("out", false, 1, 1);
  std::shared_ptr<RegstDesc> send_regst_desc = ProduceRegst("send", false, 1, 1);
  this->ForEachOutDataEdge([&](TaskEdge* edge) {
    if (edge->dst_node() == send_to_) {
      edge->AddRegst("send", send_regst_desc);
    } else {
      edge->AddRegst("out", out_regst_desc);
    }
  });
}

void RingBoxingTaskNode::ConsumeAllRegsts() {
  ConsumeRegst("recv", recv_from_->GetProducedRegst("copy_out"));
  ConsumeRegst("in", SoleInDataEdge()->GetSoleRegst());
}

void RingBoxingTaskNode::SetOutShape(const Shape& shape) { out_shape_.reset(new Shape(shape)); }

void RingBoxingTaskNode::BuildExecGphAndRegst() {
  ExecNode* node = mut_exec_gph().NewNode();
  std::shared_ptr<Operator> op = ConstructOp(GetBoxingOpConf());
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

void RingBoxingTaskNode::InferProducedDataRegstTimeShape() {
  *GetProducedRegst("out")->mut_data_regst_time_shape() =
      GetSoleConsumedRegst("in")->data_regst_time_shape();
  *GetProducedRegst("send")->mut_data_regst_time_shape() =
      GetSoleConsumedRegst("in")->data_regst_time_shape();
}

OperatorConf RingBoxingTaskNode::GetBoxingOpConf() const {
  OperatorConf op_conf{};
  op_conf.set_name("System-Boxing-RingBoxing-" + NewUniqueId());
  op_conf.set_device_type(device_type());
  RingBoxingConf* ring_boxing_conf = nullptr;
  if (mode_ == kRingBoxingTaskModeP2B) {
    ring_boxing_conf = op_conf.mutable_ring_all_reduce_conf()->mutable_ring_boxing_conf();
  } else if (mode_ == kRingBoxingTaskModeP2S) {
    ring_boxing_conf = op_conf.mutable_ring_reduce_scatter_conf()->mutable_ring_boxing_conf();
  } else if (mode_ == kRingBoxingTaskModeS2B) {
    ring_boxing_conf = op_conf.mutable_ring_all_gather_conf()->mutable_ring_boxing_conf();
  } else {
    UNIMPLEMENTED();
  }
  *ring_boxing_conf->mutable_lbi() = lbi_;
  logical_blob_shape_.ToProto(ring_boxing_conf->mutable_logical_blob_shape());
  FOR_RANGE(int64_t, i, 0, ring_.size()) {
    *ring_boxing_conf->mutable_ring()->Add() = ring_.at(i);
    slices_.at(i).ToProto(ring_boxing_conf->mutable_slices()->Add());
  }
  if (out_shape_.get() != nullptr) { out_shape_->ToProto(ring_boxing_conf->mutable_out_shape()); }
  return op_conf;
}

void RingBoxingTaskNode::ToProto(TaskProto* task_proto) {
  TaskNode::ToProto(task_proto);
  *(task_proto->mutable_parallel_ctx()) = parallel_ctx_;
}

}  // namespace oneflow
