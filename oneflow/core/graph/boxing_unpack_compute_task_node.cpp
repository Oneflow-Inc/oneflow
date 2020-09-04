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
#include "oneflow/core/graph/boxing_unpack_compute_task_node.h"
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

void BoxingUnpackCompTaskNode::Init(const CompTaskNode* src_node, const LogicalBlobId& lbi,
                                    const Shape& logical_shape, const bool need_transpose,
                                    const int64_t src_split_axis, const int64_t dst_split_axis,
                                    const int64_t parallel_num) {
  lbi_ = lbi;
  set_logical_node(src_node->logical_node());
  *mut_parallel_ctx() = *src_node->parallel_ctx();
  set_machine_id(src_node->machine_id());
  set_thrd_id(src_node->thrd_id());
  set_area_id(src_node->area_id());
  need_transpose_ = need_transpose;
  src_split_axis_ = src_split_axis;
  dst_split_axis_ = dst_split_axis;
  parallel_num_ = parallel_num;
  DimVector src_dim_vec = logical_shape.dim_vec();
  src_dim_vec[src_split_axis] = src_dim_vec.at(src_split_axis) / parallel_num;
  src_dim_vec[dst_split_axis] = src_dim_vec.at(dst_split_axis) / parallel_num;
  src_dim_vec.insert(src_dim_vec.begin(), parallel_num);
  src_shape_ = Shape(src_dim_vec);
  DimVector dst_dim_vec = logical_shape.dim_vec();
  dst_dim_vec[dst_split_axis] = dst_dim_vec.at(dst_split_axis) / parallel_num;
  dst_shape_ = Shape(dst_dim_vec);
}

void BoxingUnpackCompTaskNode::ProduceAllRegstsAndBindEdges() {
  std::shared_ptr<RegstDesc> out_regst = ProduceRegst("out", true, 1, 1);
  this->ForEachOutDataEdge([&](TaskEdge* out_dege) { out_dege->AddRegst("out", out_regst); });
}

void BoxingUnpackCompTaskNode::ConsumeAllRegsts() {
  this->ForEachInDataEdge(
      [&](TaskEdge* in_edge) { ConsumeRegst("in", SoleInDataEdge()->GetSoleRegst()); });
}

void BoxingUnpackCompTaskNode::BuildExecGphAndRegst() {
  ExecNode* node = mut_exec_gph().NewNode();
  OperatorConf op_conf;
  op_conf.set_name("System-Boxing-Unpack-" + NewUniqueId());
  op_conf.set_device_tag(CHECK_JUST(DeviceTag4DeviceType(this->device_type())));
  op_conf.mutable_boxing_unpack_conf()->set_need_transpose(need_transpose_);
  op_conf.mutable_boxing_unpack_conf()->set_src_split_axis(src_split_axis_);
  op_conf.mutable_boxing_unpack_conf()->set_dst_split_axis(dst_split_axis_);
  op_conf.mutable_boxing_unpack_conf()->set_parallel_num(parallel_num_);
  *op_conf.mutable_boxing_unpack_conf()->mutable_lbi() = lbi_;
  src_shape_.ToProto(op_conf.mutable_boxing_unpack_conf()->mutable_src_shape());
  dst_shape_.ToProto(op_conf.mutable_boxing_unpack_conf()->mutable_dst_shape());

  std::shared_ptr<Operator> sole_op = ConstructOp(op_conf, &GlobalJobDesc());
  node->mut_op() = sole_op;
  node->BindBnWithRegst(sole_op->SoleIbn(), GetSoleConsumedRegst("in"));
  std::shared_ptr<RegstDesc> out_regst = GetProducedRegst("out");
  out_regst->AddLbi(sole_op->BnInOp2Lbi(sole_op->SoleObn()));
  node->BindBnWithRegst(sole_op->SoleObn(), out_regst);
  node->InferBlobDescs(parallel_ctx());
}

void BoxingUnpackCompTaskNode::InferProducedDataRegstTimeShape() {
  NaiveInferProducedDataRegstTimeShape();
}

}  // namespace oneflow
