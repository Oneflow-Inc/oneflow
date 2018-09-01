#include "oneflow/core/graph/copy_task_node.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/job/thrd_id_generator.h"

namespace oneflow {

void CopyTaskNode::ProduceAllRegstsAndBindEdges() {
  std::string name("copy_out");
  std::shared_ptr<RegstDesc> out_regst(nullptr);
  CopyLocalTaskNode* copy_local = dynamic_cast<CopyLocalTaskNode*>(this);
  if (copy_local != nullptr) {
    TaskType dst_node_type = (*out_edges().begin())->dst_node()->GetTaskType();
    if (copy_local->copy_type() == CopyLocalOpConf::H2D
        && (dst_node_type == TaskType::kReduceLocalAdd
            || dst_node_type == TaskType::kReduceGlobalAdd
            || dst_node_type == TaskType::kReduceGather)) {
      out_regst = ProduceRegst(name, false, 1, 1);
    }
    TaskType src_node_type = SoleInEdge()->src_node()->GetTaskType();
    if (copy_local->copy_type() == CopyLocalOpConf::D2H
        && (src_node_type == TaskType::kReduceScatter || src_node_type == TaskType::kReduceLocalAdd
            || src_node_type == TaskType::kReduceGlobalAdd)) {
      out_regst = ProduceRegst(name, false, 1, 1);
    }
  }
  if (out_regst == nullptr) { out_regst = ProduceRegst(name, false); }
  for (TaskEdge* edge : out_edges()) { edge->AddRegst(name, out_regst); }
}

void CopyTaskNode::ConsumeAllRegsts() { ConsumeRegst("copy_in", SoleInEdge()->GetSoleRegst()); }

void CopyTaskNode::BuildExecGphAndRegst() {
  auto out_regst = GetProducedRegst("copy_out");
  auto in_regst = GetSoleConsumedRegst("copy_in");
  out_regst->CopyBlobDescFrom(in_regst.get());
  ExecNode* node = mut_exec_gph().NewNode();
  node->mut_op() = ConstructOp(NewCopyOpConf());
  node->BindBnWithRegst(node->op()->SoleIbn(), in_regst);
  node->BindBnWithRegst(node->op()->SoleObn(), out_regst);
}

void CopyLocalTaskNode::Init(CopyLocalOpConf::Type copy_type, int64_t machine_id,
                             int64_t dev_phy_id) {
  copy_type_ = copy_type;
  set_machine_id(machine_id);
  if (copy_type == CopyLocalOpConf::H2D) {
    set_thrd_id(Global<IDMgr>::Get()->GetGpuH2DThrdId(dev_phy_id));
  } else if (copy_type == CopyLocalOpConf::D2H) {
    set_thrd_id(Global<IDMgr>::Get()->GetGpuD2HThrdId(dev_phy_id));
  } else {
    UNIMPLEMENTED();
  }
}

void CopyLocalTaskNode::InitProducedRegstMemCase(MemoryCase* mem_case) {
  if (copy_type_ == CopyLocalOpConf::H2D) {
    TaskNode::InitProducedRegstMemCase(mem_case);
  } else if (copy_type_ == CopyLocalOpConf::D2H) {
    mem_case->mutable_host_mem()->set_used_by_device(true);
  } else {
    UNIMPLEMENTED();
  }
}

OperatorConf CopyLocalTaskNode::NewCopyOpConf() {
  OperatorConf conf;
  conf.set_name("copy_local_" + NewUniqueId());
  conf.set_device_type(device_type());
  conf.mutable_copy_local_conf()->set_type(copy_type_);
  return conf;
}

void CopyCommNetTaskNode::Init(int64_t machine_id, int64_t src_machine_id) {
  set_machine_id(machine_id);
  set_thrd_id(Global<IDMgr>::Get()->CommNetThrdId());
  peer_machine_id_ = src_machine_id;
}

namespace {

HashMap<int64_t, HashMap<int64_t, int64_t>>* GetConnection2LocalStreamIdMap() {
  // this_machine_id -> {peer_machine_id, local_work_stream_id}
  static HashMap<int64_t, HashMap<int64_t, int64_t>> connection2stream_id;
  return &connection2stream_id;
}

int64_t GetLocalStreamId4Connection(int64_t this_machine_id, int64_t peer_machine_id) {
  auto& dict = *GetConnection2LocalStreamIdMap();
  auto this_machine_it = dict.find(this_machine_id);
  if (this_machine_it == dict.end()) { return -1; }
  auto peer_machine_it = this_machine_it->second.find(peer_machine_id);
  if (peer_machine_it == this_machine_it->second.end()) { return -1; }
  return peer_machine_it->second;
}

void InsertLocalStreamId4Connection(int64_t this_machine_id, int64_t peer_machine_id) {
  auto& dict = *GetConnection2LocalStreamIdMap();
  dict[this_machine_id][peer_machine_id] = dict[this_machine_id].size();
}

}  // namespace

int64_t CopyCommNetTaskNode::AllocateLocalWorkStreamId() {
  int64_t this_machine_id = machine_id();
  int64_t local_work_stream_id = GetLocalStreamId4Connection(this_machine_id, peer_machine_id_);
  if (local_work_stream_id == -1) {
    InsertLocalStreamId4Connection(this_machine_id, peer_machine_id_);
    local_work_stream_id = GetLocalStreamId4Connection(this_machine_id, peer_machine_id_);
  }
  return local_work_stream_id;
}

void CopyCommNetTaskNode::InitProducedRegstMemCase(MemoryCase* mem_case) {
  mem_case->mutable_host_mem()->set_used_by_network(true);
}

void CopyCommNetTaskNode::PinConsumedRegstMemCase(MemoryCase* mem_case) {
  CHECK(mem_case->has_host_mem());
  mem_case->mutable_host_mem()->set_used_by_network(true);
}

OperatorConf CopyCommNetTaskNode::NewCopyOpConf() {
  OperatorConf conf;
  conf.set_name("copy_comm_net_" + NewUniqueId());
  conf.set_device_type(device_type());
  conf.mutable_copy_comm_net_conf();
  return conf;
}

}  // namespace oneflow
