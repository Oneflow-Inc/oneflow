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
#include "oneflow/core/graph/copy_task_node.h"
#include "oneflow/core/graph/task_stream_id.h"

namespace oneflow {

void CopyTaskNode::ProduceAllRegstsAndBindEdges() {
  std::shared_ptr<RegstDesc> out_regst = ProduceRegst("copy_out", false);
  ForEachOutDataEdge([&](TaskEdge* edge) { edge->AddRegst("copy_out", out_regst); });
}

void CopyTaskNode::ConsumeAllRegsts() { ConsumeRegst("copy_in", SoleInDataEdge()->GetSoleRegst()); }

void CopyTaskNode::BuildExecGphAndRegst() {
  auto out_regst = GetProducedRegst("copy_out");
  auto in_regst = GetSoleConsumedRegst("copy_in");
  out_regst->CopyBlobDescFrom(in_regst.get());
  ExecNode* node = mut_exec_gph().NewNode();
  node->mut_op() = CHECK_JUST(ConstructOp(NewCopyOpConf()));
  node->BindBnWithRegst(node->op()->SoleIbn(), in_regst);
  node->BindBnWithRegst(node->op()->SoleObn(), out_regst);
}

void CopyTaskNode::InferProducedDataRegstTimeShape() { NaiveInferProducedDataRegstTimeShape(); }

void CopyHdTaskNode::Init(CopyHdOpConf::Type copy_type, const DeviceId& device_id,
                          const LogicalBlobId& lbi) {
  copy_type_ = copy_type;
  set_machine_id(device_id.rank());
  int64_t thrd_id = -1;
  if (copy_type == CopyHdOpConf::H2D) {
    thrd_id = EncodeStreamIdToInt64(GenerateNamedTaskStreamId(device_id, "H2D"));
  } else if (copy_type == CopyHdOpConf::D2H) {
    thrd_id = EncodeStreamIdToInt64(GenerateNamedTaskStreamId(device_id, "D2H"));
  } else {
    UNIMPLEMENTED();
  }
  set_thrd_id(thrd_id);
  set_lbi(lbi);
}

void CopyHdTaskNode::InitProducedRegstMemCase(MemoryCase* mem_case) {
  if (copy_type_ == CopyHdOpConf::H2D) {
    TaskNode::InitProducedRegstMemCase(mem_case);
  } else if (copy_type_ == CopyHdOpConf::D2H) {
    mem_case->set_device_type(DeviceType::kCPU);
    mem_case->set_device_id(0);
    mem_case->set_pinned_device_type(device_type());
    mem_case->set_pinned_device_id(stream_id().device_id().device_index());
  } else {
    UNIMPLEMENTED();
  }
}

OperatorConf CopyHdTaskNode::NewCopyOpConf() {
  OperatorConf conf;
  conf.set_name("copy_hd_" + NewUniqueId());
  conf.set_device_tag(*CHECK_JUST(DeviceTag4DeviceType(device_type())));
  conf.mutable_copy_hd_conf()->set_type(copy_type_);
  auto in_regst = GetSoleConsumedRegst("copy_in");
  CHECK_EQ(in_regst->NumOfLbi(), 1);
  in_regst->ForEachLbi([&](const LogicalBlobId& lbi) {
    *conf.mutable_copy_hd_conf()->mutable_lbi() = lbi;
    CHECK(lbi == this->lbi());
  });
  return conf;
}

void CopyCommNetTaskNode::Init(int64_t machine_id, const LogicalBlobId& lbi) {
  set_machine_id(machine_id);
  set_thrd_id(EncodeStreamIdToInt64(
      GenerateNamedTaskStreamId(machine_id, DeviceType::kCPU, 0, "COMM_NET")));
  set_lbi(lbi);
}

OperatorConf CopyCommNetTaskNode::NewCopyOpConf() {
  OperatorConf conf;
  conf.set_name("copy_comm_net_" + NewUniqueId());
  conf.set_device_tag(*CHECK_JUST(DeviceTag4DeviceType(this->device_type())));
  *(conf.mutable_copy_comm_net_conf()->mutable_lbi()) = lbi();
  return conf;
}

}  // namespace oneflow
