/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include "oneflow/core/framework/to_string.h"
#include "oneflow/core/graph/copy_task_node.h"
#include "oneflow/core/job/thrd_id_generator.h"
#include "oneflow/core/operator/operator.h"

namespace oneflow {

void CopyTaskNode::ProduceAllRegstsAndBindEdges() {
  std::string name("copy_out");
  std::shared_ptr<RegstDesc> out_regst(nullptr);
  CopyHdTaskNode* copy_hd = dynamic_cast<CopyHdTaskNode*>(this);
  if (copy_hd != nullptr) {
    TaskNode* first_dst_node = nullptr;
    ForEachNodeOnOutDataEdge([&](TaskNode* node) {
      if (first_dst_node == nullptr) { first_dst_node = node; }
    });
    if (out_regst == nullptr) {
      // normal copy hd task can reuse mem
      out_regst = ProduceRegst(name, true);
    }
  }
  if (out_regst == nullptr) {
    // copy comm_net task cannot reuse mem
    out_regst = ProduceRegst(name, false);
  }
  ForEachOutDataEdge([&](TaskEdge* edge) { edge->AddRegst(name, out_regst); });
}

void CopyTaskNode::ConsumeAllRegsts() { ConsumeRegst("copy_in", SoleInDataEdge()->GetSoleRegst()); }

void CopyTaskNode::BuildExecGphAndRegst() {
  auto out_regst = GetProducedRegst("copy_out");
  auto in_regst = GetSoleConsumedRegst("copy_in");
  out_regst->CopyBlobDescFrom(in_regst.get());
  ExecNode* node = mut_exec_gph().NewNode();
  node->mut_op() = ConstructOp(NewCopyOpConf(), &GlobalJobDesc());
  node->BindBnWithRegst(node->op()->SoleIbn(), in_regst);
  node->BindBnWithRegst(node->op()->SoleObn(), out_regst);
}

void CopyTaskNode::InferProducedDataRegstTimeShape() { NaiveInferProducedDataRegstTimeShape(); }

void CopyHdTaskNode::Init(CopyHdOpConf::Type copy_type, int64_t machine_id, int64_t dev_phy_id) {
  copy_type_ = copy_type;
  set_machine_id(machine_id);
  if (copy_type == CopyHdOpConf::H2D) {
    set_thrd_id(Global<IDMgr>::Get()->GetGpuH2DThrdId(dev_phy_id));
  } else if (copy_type == CopyHdOpConf::D2H) {
    set_thrd_id(Global<IDMgr>::Get()->GetGpuD2HThrdId(dev_phy_id));
  } else {
    UNIMPLEMENTED();
  }
}

void CopyHdTaskNode::InitProducedRegstMemCase(MemoryCase* mem_case) {
  if (copy_type_ == CopyHdOpConf::H2D) {
    TaskNode::InitProducedRegstMemCase(mem_case);
  } else if (copy_type_ == CopyHdOpConf::D2H) {
    mem_case->mutable_host_mem()->mutable_cuda_pinned_mem()->set_device_id(GpuPhyId());
  } else {
    UNIMPLEMENTED();
  }
}

OperatorConf CopyHdTaskNode::NewCopyOpConf() {
  OperatorConf conf;
  conf.set_name("copy_hd_" + NewUniqueId());
  const char* device_tag = CHECK_JUST(DeviceTag4DeviceType(device_type()));
  conf.set_device_tag(device_tag);
  conf.mutable_copy_hd_conf()->set_type(copy_type_);
  auto in_regst = GetSoleConsumedRegst("copy_in");
  if (in_regst->NumOfLbi() == 1) {
    in_regst->ForEachLbi(
        [&](const LogicalBlobId& lbi) { *conf.mutable_copy_hd_conf()->mutable_lbi() = lbi; });
  }
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
  const char* device_tag = CHECK_JUST(DeviceTag4DeviceType(this->device_type()));
  conf.set_device_tag(device_tag);
  conf.mutable_copy_comm_net_conf();
  return conf;
}

}  // namespace oneflow
