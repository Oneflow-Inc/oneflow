#include "oneflow/core/graph/cuda_ring_all_reduce_task_node.h"

namespace oneflow {

void CudaRingAllReduceTaskNode::Init(int64_t machine_id, int64_t thrd_id, const LogicalBlobId& lbi,
                                     const Shape& logical_blob_shape,
                                     const ParallelContext& parallel_ctx) {
  set_machine_id(machine_id);
  set_thrd_id(thrd_id);
  set_area_id(kMdUpdtArea);
  lbi_ = lbi;
  logical_blob_shape_ = logical_blob_shape;
  parallel_ctx_ = parallel_ctx;
}

bool CudaRingAllReduceTaskNode::IsReadyForBuild() { return GetSoleConsumedRegst("in")->IsLocked(); }

void CudaRingAllReduceTaskNode::ProduceAllRegstsAndBindEdges() {
  std::shared_ptr<RegstDesc> out_regst_desc = ProduceRegst("out", false, 1, 1);
  HashSet<TaskNode*> send_to_nodes;
  FOR_RANGE(int64_t, i, 0, send_to_.size()) {
    std::shared_ptr<RegstDesc> send_regst_desc =
        ProduceRegst("send_" + std::to_string(i), false, 2, 2);
    send_regst_desc->mut_mem_case()->mutable_host_mem()->set_used_by_device_id(GpuPhyId());
    send_to_nodes.emplace(send_to_.at(i));
  }
  this->ForEachOutDataEdge([&](TaskEdge* edge) {
    auto it = send_to_nodes.find(edge->dst_node());
    if (it == send_to_nodes.cend()) {
      edge->AddRegst("out", out_regst_desc);
    } else {
      UNIMPLEMENTED();
    }
  });
}

void CudaRingAllReduceTaskNode::ConsumeAllRegsts() {
  FOR_RANGE(int64_t, i, 0, recv_from_.size()) {
    if (dynamic_cast<CudaRingAllReduceTaskNode*>(recv_from_.at(i)) != nullptr) {
      ConsumeRegst("recv_" + std::to_string(i),
                   recv_from_.at(i)->GetProducedRegst("send_" + std::to_string(i)));
    } else {
      UNIMPLEMENTED();
    }
  }
  ConsumeRegst("in", SoleInDataEdge()->GetSoleRegst());
}

void CudaRingAllReduceTaskNode::AddRing(const std::vector<int64_t>& ring_next, TaskNode* send_to,
                                        TaskNode* recv_from) {
  rings_.push_back(ring_next);
  send_to_.push_back(send_to);
  recv_from_.push_back(recv_from);
}

void CudaRingAllReduceTaskNode::BuildExecGphAndRegst() {
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

void CudaRingAllReduceTaskNode::InferProducedDataRegstTimeShape() {
  *GetProducedRegst("out")->mut_data_regst_time_shape() =
      GetSoleConsumedRegst("in")->data_regst_time_shape();
  FOR_RANGE(int64_t, i, 0, send_to_.size()) {
    *GetProducedRegst("send_" + std::to_string(i))->mut_data_regst_time_shape() =
        GetSoleConsumedRegst("in")->data_regst_time_shape();
  }
}

OperatorConf CudaRingAllReduceTaskNode::GenOpConf() const {
  OperatorConf op_conf{};
  op_conf.set_name("System-Boxing-CudaRingAllReduce-" + NewUniqueId());
  op_conf.set_device_type(device_type());
  CudaRingAllReduceOpConf* all_reduce_conf = op_conf.mutable_cuda_ring_all_reduce_conf();
  *all_reduce_conf->mutable_lbi() = lbi_;
  FOR_RANGE(int64_t, ring_id, 0, rings_.size()) {
    RingLinkConf link_conf{};
    for (const int64_t id : rings_.at(ring_id)) { link_conf.mutable_next()->Add(id); }
    *all_reduce_conf->mutable_link()->Add() = link_conf;
  }
  logical_blob_shape_.ToProto(all_reduce_conf->mutable_logical_blob_shape());
  return op_conf;
}

void CudaRingAllReduceTaskNode::ToProto(TaskProto* task_proto) {
  TaskNode::ToProto(task_proto);
  *(task_proto->mutable_parallel_ctx()) = parallel_ctx_;
}

void CudaRingAllReduceTaskNode::PinConsumedRegstMemCase(MemoryCase* mem_case) {
  if (mem_case->has_host_mem()) { mem_case->mutable_host_mem()->set_used_by_device_id(GpuPhyId()); }
}

}  // namespace oneflow
