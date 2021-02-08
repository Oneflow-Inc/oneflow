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
#include "oneflow/core/graph/collective_boxing_pack_task_node.h"

namespace oneflow {

void CollectiveBoxingPackTaskNode::Init(int64_t machine_id, int64_t thrd_id, int64_t area_id,
                                        const LogicalBlobId& lbi, const Shape& logical_shape,
                                        const SbpParallel& src_sbp_parallel,
                                        const SbpParallel& dst_sbp_parallel,
                                        const int64_t parallel_num) {
  lbi_ = lbi;
  set_machine_id(machine_id);
  set_thrd_id(thrd_id);
  set_area_id(area_id);
  logical_shape_ = logical_shape;
  parallel_num_ = parallel_num;
  src_sbp_parallel_ = src_sbp_parallel;
  dst_sbp_parallel_ = dst_sbp_parallel;
}

void CollectiveBoxingPackTaskNode::ProduceAllRegstsAndBindEdges() {
  std::shared_ptr<RegstDesc> out_regst = ProduceRegst("out", true, 1, 1);
  this->ForEachOutDataEdge([&](TaskEdge* out_dege) { out_dege->AddRegst("out", out_regst); });
}

void CollectiveBoxingPackTaskNode::ConsumeAllRegsts() {
  this->ForEachInDataEdge(
      [&](TaskEdge* in_edge) { ConsumeRegst("in", SoleInDataEdge()->GetSoleRegst()); });
}

void CollectiveBoxingPackTaskNode::BuildExecGphAndRegst() {
  ExecNode* node = mut_exec_gph().NewNode();
  OperatorConf op_conf;
  op_conf.set_name("System-Collective-Boxing-Pack-" + NewUniqueId());
  op_conf.set_device_tag(CHECK_JUST(DeviceTag4DeviceType(this->device_type())));
  auto* collective_boxing_pack_conf = op_conf.mutable_collective_boxing_pack_conf();
  *collective_boxing_pack_conf->mutable_lbi() = lbi_;
  logical_shape_.ToProto(collective_boxing_pack_conf->mutable_logical_shape());
  *collective_boxing_pack_conf->mutable_src_sbp_parallel() = src_sbp_parallel_;
  *collective_boxing_pack_conf->mutable_dst_sbp_parallel() = dst_sbp_parallel_;
  collective_boxing_pack_conf->set_num_ranks(parallel_num_);
  std::shared_ptr<Operator> sole_op = ConstructOp(op_conf, &GlobalJobDesc());
  node->mut_op() = sole_op;
  node->BindBnWithRegst(sole_op->SoleIbn(), GetSoleConsumedRegst("in"));
  std::shared_ptr<RegstDesc> out_regst = GetProducedRegst("out");
  out_regst->AddLbi(sole_op->BnInOp2Lbi(sole_op->SoleObn()));
  node->BindBnWithRegst(sole_op->SoleObn(), out_regst);
  node->InferBlobDescs(nullptr);
}

void CollectiveBoxingPackTaskNode::InferProducedDataRegstTimeShape() {
  NaiveInferProducedDataRegstTimeShape();
}

}  // namespace oneflow
