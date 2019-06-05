#include "oneflow/core/graph/slice_boxing_task_node.h"

namespace oneflow {

void SliceBoxingTaskNode::Init(const LogicalBlobId& lbi, const TensorSliceView& out_slice,
                               const SliceBoxingTaskMode mode, int64_t machine_id, int64_t thrd_id,
                               const MemoryCase& mem_case) {
  lbi_ = lbi;
  out_slice_ = out_slice;
  mode_ = mode;
  mem_case_ = mem_case;
  set_machine_id(machine_id);
  set_thrd_id(thrd_id);
  set_area_id(kMdUpdtArea);
}

void SliceBoxingTaskNode::Init(const LogicalBlobId& lbi, const TensorSliceView& out_slice,
                               const SliceBoxingTaskMode mode, int64_t machine_id,
                               int64_t thrd_id) {
  IDMgr* global_id_mgr = Global<IDMgr>::Get();
  DeviceType device_type = global_id_mgr->GetDeviceTypeFromThrdId(thrd_id);
  MemoryCase memory_case;
  if (device_type == DeviceType::kCPU) {
    memory_case.mutable_host_mem();
  } else if (device_type == DeviceType::kGPU) {
    memory_case.mutable_device_cuda_mem()->set_device_id(
        global_id_mgr->GetGpuPhyIdFromThrdId(thrd_id));
  } else {
    UNIMPLEMENTED();
  }
  Init(lbi, out_slice, mode, machine_id, thrd_id, memory_case);
}

void SliceBoxingTaskNode::ProduceAllRegstsAndBindEdges() {
  std::shared_ptr<RegstDesc> out_regst_desc = ProduceRegst("out", false, 1, 1);
  this->ForEachOutDataEdge([&](TaskEdge* edge) { edge->AddRegst("out", out_regst_desc); });
  ProduceRegst("fw_buf", false, 1, 1);
}

void SliceBoxingTaskNode::ConsumeAllRegsts() {
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

void SliceBoxingTaskNode::BuildExecGphAndRegst() {
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

void SliceBoxingTaskNode::InferProducedDataRegstTimeShape() {
  NaiveInferProducedDataRegstTimeShape();
}

void SliceBoxingTaskNode::SetInDataEdgeSlice(const TaskEdge* edge, const TensorSliceView& slice) {
  CHECK(in_data_edge2slice_.emplace(edge, slice).second);
  ordered_in_data_edges_.push_back(edge);
}

void SliceBoxingTaskNode::ConnectToSrcNodeWithSlice(TaskNode* src, TaskEdge* edge,
                                                    const TensorSliceView& slice) {
  Connect<TaskNode>(src, edge, this);
  SetInDataEdgeSlice(edge, slice);
}

OperatorConf SliceBoxingTaskNode::GetBoxingOpConf() {
  OperatorConf op_conf{};
  op_conf.set_device_type(device_type());
  SliceBoxingConf boxing_conf{};
  *boxing_conf.mutable_lbi() = lbi_;
  out_slice_.ToProto(boxing_conf.mutable_out_slice());
  for (const TaskEdge* edge : ordered_in_data_edges_) {
    in_data_edge2slice_.at(edge).ToProto(boxing_conf.mutable_in_slice()->Add());
  }
  if (mode_ == kSliceBoxingTaskModeCopy) {
    op_conf.set_name("System-Boxing-BoxingCopy-" + NewUniqueId());
    SliceBoxingCopyOpConf* conf = op_conf.mutable_slice_boxing_copy_conf();
    *conf->mutable_slice_boxing_conf() = boxing_conf;
  } else if (mode_ == kSliceBoxingTaskModeAdd) {
    op_conf.set_name("System-Boxing-BoxingAdd-" + NewUniqueId());
    SliceBoxingAddOpConf* conf = op_conf.mutable_slice_boxing_add_conf();
    *conf->mutable_slice_boxing_conf() = boxing_conf;
  } else {
    UNIMPLEMENTED();
  }
  return op_conf;
}

void SliceBoxingTaskNode::InitProducedRegstMemCase(MemoryCase* mem_case) {
  *mem_case = mem_case_;
  if (mem_case_.has_host_mem() && device_type() == DeviceType::kGPU) {
    mem_case->mutable_host_mem()->set_used_by_device(true);
  }
}

}  // namespace oneflow
