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
#include "oneflow/core/graph/boxing_zeros_task_node.h"

namespace oneflow {

void BoxingZerosTaskNode::Init(int64_t machine_id, int64_t thrd_id, const LogicalBlobId& lbi,
                               const Shape& shape, DataType data_type, const Shape& time_shape) {
  lbi_ = lbi;
  set_machine_id(machine_id);
  set_thrd_id(thrd_id);
  shape_ = shape;
  data_type_ = data_type;
  time_shape_ = time_shape;
}

void BoxingZerosTaskNode::ProduceAllRegstsAndBindEdges() {
  std::shared_ptr<RegstDesc> out_regst = ProduceRegst("out", false, 1, 1);
  this->ForEachOutDataEdge([&](TaskEdge* out_dege) { out_dege->AddRegst("out", out_regst); });
}

void BoxingZerosTaskNode::ConsumeAllRegsts() {
  // do nothing
}

void BoxingZerosTaskNode::BuildExecGphAndRegst() {
  ExecNode* node = mut_exec_gph().NewNode();
  OperatorConf op_conf;
  op_conf.set_name("System-Boxing-Zeros-" + NewUniqueId());
  op_conf.set_device_tag(CHECK_JUST(DeviceTag4DeviceType(this->device_type())));
  *op_conf.mutable_boxing_zeros_conf()->mutable_lbi() = lbi_;
  shape_.ToProto(op_conf.mutable_boxing_zeros_conf()->mutable_shape());
  op_conf.mutable_boxing_zeros_conf()->set_data_type(data_type_);
  std::shared_ptr<Operator> sole_op = ConstructOp(op_conf, &GlobalJobDesc());
  node->mut_op() = sole_op;
  std::shared_ptr<RegstDesc> out_regst = GetProducedRegst("out");
  out_regst->AddLbi(sole_op->BnInOp2Lbi(sole_op->SoleObn()));
  node->BindBnWithRegst(sole_op->SoleObn(), out_regst);
  node->InferBlobDescs(nullptr);
}

void BoxingZerosTaskNode::InferProducedDataRegstTimeShape() {
  GetProducedRegst("out")->mut_data_regst_time_shape()->reset(new Shape(time_shape_));
}

}  // namespace oneflow
