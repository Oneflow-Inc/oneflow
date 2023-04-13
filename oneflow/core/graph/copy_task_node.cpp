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
#include "oneflow/core/framework/user_op_registry_manager.h"

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
  auto constructed = CHECK_JUST(ConstructOp(NewCopyOpConf()));

  // prevent filling parallel desc for copy commnet
  if (constructed->op_conf().has_user_conf()) {
    std::shared_ptr<Shape> hierarchy = std::make_shared<Shape>(Shape({1}));
    auto parallel_desc =
        ParallelDesc::New(constructed->op_conf().device_tag(), {"0:0-0"}, hierarchy).GetOrThrow();
    CHECK_JUST(constructed->FillOpParallelDesc(parallel_desc));
  }

  node->mut_op() = constructed;
  node->BindBnWithRegst(node->op()->SoleIbn(), in_regst);
  node->BindBnWithRegst(node->op()->SoleObn(), out_regst);
}

void CopyTaskNode::InferProducedDataRegstTimeShape() { NaiveInferProducedDataRegstTimeShape(); }

void CopyHdTaskNode::Init(CopyHdType copy_type, const DeviceId& device_id,
                          const LogicalBlobId& lbi) {
  copy_type_ = copy_type;
  set_machine_id(device_id.rank());
  int64_t thrd_id = -1;
  if (copy_type == CopyHdType::H2D) {
    thrd_id = EncodeStreamIdToInt64(GenerateNamedTaskStreamId(device_id, "H2D"));
  } else if (copy_type == CopyHdType::D2H) {
    thrd_id = EncodeStreamIdToInt64(GenerateNamedTaskStreamId(device_id, "D2H"));
  } else {
    UNIMPLEMENTED();
  }
  set_thrd_id(thrd_id);
  set_lbi(lbi);
}

void CopyHdTaskNode::InitProducedRegstMemCase(MemoryCase* mem_case) {
  if (copy_type_ == CopyHdType::H2D) {
    TaskNode::InitProducedRegstMemCase(mem_case);
  } else if (copy_type_ == CopyHdType::D2H) {
    mem_case->set_device_type(DeviceType::kCPU);
    mem_case->set_device_id(0);
    mem_case->set_pinned_device_type(device_type());
    mem_case->set_pinned_device_id(stream_id().device_id().device_index());
  } else {
    UNIMPLEMENTED();
  }
}

void CopyHdTaskNode::ProduceAllRegstsAndBindEdges() {
  const bool enable_mem_reuse = ParseBooleanFromEnv("ONEFLOW_GRAPH_BOXING_ENABLE_MEM_REUSE", false)
                                && (copy_type_ == CopyHdType::H2D);
  std::shared_ptr<RegstDesc> out_regst = ProduceRegst("copy_out", enable_mem_reuse);
  ForEachOutDataEdge([&](TaskEdge* edge) { edge->AddRegst("copy_out", out_regst); });
}

OperatorConf CopyHdTaskNode::NewCopyOpConf() {
  OperatorConf conf;
  conf.set_device_tag(*CHECK_JUST(DeviceTag4DeviceType(device_type())));
  auto copy_type_name = "undefined";
  if (copy_type_ == CopyHdType::D2H) {
    copy_type_name = "copy_d2h";
  } else if (copy_type_ == CopyHdType::H2D) {
    copy_type_name = "copy_h2d";
  } else {
    LOG(FATAL) << "unknow copy type: " << copy_type_;
  }
  conf.set_name(std::string(copy_type_name) + "_" + NewUniqueId());
  *conf.mutable_user_conf()->mutable_op_type_name() = copy_type_name;
  auto in_regst = GetSoleConsumedRegst("copy_in");
  CHECK_EQ(in_regst->NumOfLbi(), 1);
  in_regst->ForEachLbi([&](const LogicalBlobId& lbi) {
    (*conf.mutable_user_conf()->mutable_input())["in"].add_s(GenLogicalBlobName(lbi));
    (*conf.mutable_user_conf()->mutable_output())["out"].add_s(
        GenLogicalBlobName(conf.name(), GenRepeatedBn("out", 0)));
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
