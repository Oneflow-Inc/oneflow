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
#include "oneflow/core/graph/slice_boxing_task_node.h"

#include "oneflow/core/framework/to_string.h"
#include "oneflow/core/graph/task_graph_rebuild_ctx.h"

namespace oneflow {

void SliceBoxingTaskNode::Init(const LogicalBlobId& lbi, const TensorSliceView& out_slice,
                               const SliceBoxingTaskMode mode, int64_t machine_id,
                               int64_t thrd_id) {
  out_slice_ = out_slice;
  out_shape_ = out_slice.shape();
  mode_ = mode;
  set_machine_id(machine_id);
  set_thrd_id(thrd_id);
  set_lbi(lbi);
}

void SliceBoxingTaskNode::ProduceAllRegstsAndBindEdges() {
  std::shared_ptr<RegstDesc> out_regst_desc = ProduceRegst("out", true);
  this->ForEachOutDataEdge([&](TaskEdge* edge) { edge->AddRegst("out", out_regst_desc); });
  ProduceRegst("tmp", true);
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
  std::shared_ptr<Operator> op = CHECK_JUST(ConstructOp(GetBoxingOpConf()));
  node->mut_op() = op;
  FOR_RANGE(size_t, i, 0, op->input_bns().size()) {
    const std::string& ibn = op->input_bns().Get(i);
    CHECK_EQ(GenUnRepeatedBn(ibn).second, i);
    node->BindBnWithRegst(ibn, GetSoleConsumedRegst("in_" + std::to_string(i)));
  }
  std::shared_ptr<RegstDesc> out_regst = GetProducedRegst("out");
  out_regst->AddLbi(lbi());
  node->BindBnWithRegst(op->SoleObn(), out_regst);
  node->AddBnToRegstAndBindIt(&Operator::tmp_bns, GetProducedRegst("tmp"));
  (node->*GetInferBlobDescsMethod())(parallel_ctx());
}

void SliceBoxingTaskNode::InferProducedDataRegstTimeShape() {
  NaiveInferProducedDataRegstTimeShape();
}

void SliceBoxingTaskNode::SetInDataEdgeSlice(const TaskEdge* edge, const TensorSliceView& slice) {
  CHECK(in_data_edge2slice_.emplace(edge, slice).second);
  ordered_in_data_edges_.emplace_back(edge);
}

void SliceBoxingTaskNode::ConnectToSrcNodeWithSlice(TaskNode* src, TaskEdge* edge,
                                                    const TensorSliceView& slice) {
  edge->AddLbi(lbi());
  Connect<TaskNode>(src, edge, this);
  SetInDataEdgeSlice(edge, slice);
}

void SliceBoxingTaskNode::SetOutShape(const Shape& shape) { out_shape_ = shape; }

OperatorConf SliceBoxingTaskNode::GetBoxingOpConf() {
  OperatorConf op_conf{};
  op_conf.set_device_tag(*CHECK_JUST(DeviceTag4DeviceType(device_type())));
  SliceBoxingConf boxing_conf{};
  *boxing_conf.mutable_lbi() = lbi();
  out_slice_.ToProto(boxing_conf.mutable_out_slice());
  out_shape_.ToProto(boxing_conf.mutable_out_shape());
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

Maybe<void> SliceBoxingTaskNode::InitTransportTaskFromProto(
    const TransportTaskProto& transport_task_proto, const TaskGraphRebuildCtx& ctx) {
  CHECK_OR_RETURN(transport_task_proto.has_slice_boxing_task())
      << "not a serialized SliceBoxingTaskNode. debug string: "
      << transport_task_proto.DebugString();
  const auto& proto = transport_task_proto.slice_boxing_task();
  for (const auto& pair : proto.in_data_edge_uid2slice()) {
    const auto* edge = JUST(ctx.TaskEdge4Uid(pair.first));
    CHECK_OR_RETURN(in_data_edge2slice_.emplace(edge, pair.second).second)
        << "redundant edge found. edge_uid: " << pair.first;
  }
  for (int64_t edge_uid : proto.ordered_in_data_edge_uid()) {
    ordered_in_data_edges_.push_back(JUST(ctx.TaskEdge4Uid(edge_uid)));
  }
  out_slice_ = TensorSliceView(proto.out_slice());
  out_shape_ = Shape(proto.out_shape());
  mode_ = proto.mode();
  return Maybe<void>::Ok();
}

void SliceBoxingTaskNode::ToTransportTaskProto(TransportTaskProto* transport_task_proto) const {
  ToProto(transport_task_proto->mutable_task_proto(), /*check=*/false);
  auto* proto = transport_task_proto->mutable_slice_boxing_task();
  for (const auto& pair : in_data_edge2slice_) {
    int64_t edge_uid = reinterpret_cast<int64_t>(pair.first);
    pair.second.ToProto(&(*proto->mutable_in_data_edge_uid2slice())[edge_uid]);
  }
  for (const auto* edge : ordered_in_data_edges_) {
    proto->add_ordered_in_data_edge_uid(reinterpret_cast<int64_t>(edge));
  }
  out_slice_.ToProto(proto->mutable_out_slice());
  out_shape_.ToProto(proto->mutable_out_shape());
  proto->set_mode(mode_);
}

}  // namespace oneflow
