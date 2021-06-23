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
#include "oneflow/core/graph/boxing_identity_task_node.h"

namespace oneflow {

void BoxingIdentityTaskNode::Init(int64_t machine_id, int64_t thrd_id, const LogicalBlobId& lbi) {
  set_machine_id(machine_id);
  set_thrd_id(thrd_id);
  set_lbi(lbi);
}

void BoxingIdentityTaskNode::ProduceAllRegstsAndBindEdges() {
  std::shared_ptr<RegstDesc> out_regst = ProduceRegst("out", true, 1, 1);
  this->ForEachOutDataEdge([&](TaskEdge* out_dege) { out_dege->AddRegst("out", out_regst); });
}

void BoxingIdentityTaskNode::ConsumeAllRegsts() {
  this->ForEachInDataEdge(
      [&](TaskEdge* in_edge) { ConsumeRegst("in", SoleInDataEdge()->GetSoleRegst()); });
}

void BoxingIdentityTaskNode::BuildExecGphAndRegst() {
  ExecNode* node = mut_exec_gph().NewNode();
  OperatorConf op_conf;
  op_conf.set_name("System-Boxing-Identity-" + NewUniqueId());
  op_conf.set_device_tag(*CHECK_JUST(DeviceTag4DeviceType(this->device_type())));
  *op_conf.mutable_boxing_identity_conf()->mutable_lbi() = lbi();
  std::shared_ptr<Operator> sole_op = ConstructOp(op_conf);
  node->mut_op() = sole_op;
  node->BindBnWithRegst(sole_op->SoleIbn(), GetSoleConsumedRegst("in"));
  std::shared_ptr<RegstDesc> out_regst = GetProducedRegst("out");
  out_regst->AddLbi(sole_op->BnInOp2Lbi(sole_op->SoleObn()));
  node->BindBnWithRegst(sole_op->SoleObn(), out_regst);
  node->InferBlobDescs(nullptr);
}

void BoxingIdentityTaskNode::InferProducedDataRegstTimeShape() {
  NaiveInferProducedDataRegstTimeShape();
}

}  // namespace oneflow
