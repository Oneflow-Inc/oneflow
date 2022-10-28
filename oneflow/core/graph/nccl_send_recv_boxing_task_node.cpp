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
#include "oneflow/core/graph/nccl_send_recv_boxing_task_node.h"

namespace oneflow {

void NcclSendRecvBoxingTaskNode::Init(int64_t machine_id, int64_t thrd_id, const LogicalBlobId& lbi,
                                      const Shape& logical_shape, const DataType& data_type,
                                      const NdSbp& src_nd_sbp, const NdSbp& dst_nd_sbp,
                                      const ParallelDesc& src_parallel_desc,
                                      const ParallelDesc& dst_parallel_desc,
                                      const int64_t parallel_id, const ParallelDesc& parallel_desc,
                                      const bool has_input, const bool has_output,
                                      const std::string& stream_name) {
  set_machine_id(machine_id);
  set_thrd_id(thrd_id);
  set_lbi(lbi);
  logical_shape_ = logical_shape;
  src_nd_sbp_ = src_nd_sbp;
  dst_nd_sbp_ = dst_nd_sbp;
  src_parallel_conf_ = src_parallel_desc.parallel_conf();
  dst_parallel_conf_ = dst_parallel_desc.parallel_conf();
  parallel_conf_ = parallel_desc.parallel_conf();
  parallel_ctx_.set_parallel_id(parallel_id);
  parallel_ctx_.set_parallel_num(parallel_desc.parallel_num());
  has_input_ = has_input;
  has_output_ = has_output;
  data_type_ = data_type;
  stream_name_ = stream_name;
}

void NcclSendRecvBoxingTaskNode::ProduceAllRegstsAndBindEdges() {
  if (has_output_) {
    std::shared_ptr<RegstDesc> out_regst = ProduceRegst("out", true, 1, 1);
    this->ForEachOutDataEdge([&](TaskEdge* out_dege) { out_dege->AddRegst("out", out_regst); });
  }
  ProduceRegst("tmp", true);
}

void NcclSendRecvBoxingTaskNode::ConsumeAllRegsts() {
  this->ForEachInDataEdge(
      [&](TaskEdge* in_edge) { ConsumeRegst("in", SoleInDataEdge()->GetSoleRegst()); });
}

void NcclSendRecvBoxingTaskNode::BuildExecGphAndRegst() {
  ExecNode* node = mut_exec_gph().NewNode();
  OperatorConf op_conf;
  op_conf.set_name("System-Nccl-Send-Recv-Boxing-" + NewUniqueId());
  op_conf.set_device_tag(*CHECK_JUST(DeviceTag4DeviceType(this->device_type())));
  op_conf.set_stream_name_hint(stream_name_);
  auto* nccl_send_recv_boxing_conf = op_conf.mutable_nccl_send_recv_boxing_conf();
  *nccl_send_recv_boxing_conf->mutable_lbi() = lbi();
  logical_shape_.ToProto(nccl_send_recv_boxing_conf->mutable_logical_shape());
  nccl_send_recv_boxing_conf->set_data_type(data_type_);
  *nccl_send_recv_boxing_conf->mutable_src_nd_sbp() = src_nd_sbp_;
  *nccl_send_recv_boxing_conf->mutable_dst_nd_sbp() = dst_nd_sbp_;
  *nccl_send_recv_boxing_conf->mutable_parallel_conf() = parallel_conf_;
  *nccl_send_recv_boxing_conf->mutable_src_parallel_conf() = src_parallel_conf_;
  *nccl_send_recv_boxing_conf->mutable_dst_parallel_conf() = dst_parallel_conf_;
  nccl_send_recv_boxing_conf->set_has_input(has_input_);
  nccl_send_recv_boxing_conf->set_has_output(has_output_);
  std::shared_ptr<Operator> sole_op = CHECK_JUST(ConstructOp(op_conf));
  node->mut_op() = sole_op;
  CHECK_JUST(sole_op->FillOpParallelDesc(parallel_conf_));
  if (has_input_) { node->BindBnWithRegst(sole_op->SoleIbn(), GetSoleConsumedRegst("in")); }
  if (has_output_) {
    std::shared_ptr<RegstDesc> out_regst = GetProducedRegst("out");
    out_regst->AddLbi(sole_op->BnInOp2Lbi(sole_op->SoleObn()));
    node->BindBnWithRegst(sole_op->SoleObn(), out_regst);
  }
  node->AddBnToRegstAndBindIt(&Operator::tmp_bns, GetProducedRegst("tmp"));
  node->InferBlobDescs(parallel_ctx());
}

void NcclSendRecvBoxingTaskNode::InferProducedDataRegstTimeShape() {
  auto out_regst = GetProducedRegst("out");
  if (out_regst != nullptr) { out_regst->mut_data_regst_time_shape()->reset(new Shape({1, 1})); }
  auto tmp_regst = GetProducedRegst("tmp");
  tmp_regst->mut_data_regst_time_shape()->reset(new Shape({1, 1}));
}

}  // namespace oneflow
