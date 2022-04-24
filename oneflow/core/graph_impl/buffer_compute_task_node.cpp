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
#include "oneflow/core/graph/compute_task_node.h"
#include "oneflow/core/graph/task_stream_index_manager.h"
#include "oneflow/core/framework/framework.h"

namespace oneflow {

class BufferCompTaskNode final : public CompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BufferCompTaskNode);
  BufferCompTaskNode() = default;
  ~BufferCompTaskNode() = default;
  TaskType GetTaskType() const override { return TaskType::kBuffer; }
  void BuildExecGphAndRegst() override;
  void ProduceAllRegstsAndBindEdges() override;
  void ConsumeAllRegsts() override;
};

void BufferCompTaskNode::ProduceAllRegstsAndBindEdges() {
  std::shared_ptr<const Operator> sole_op = op();
  CHECK(sole_op->op_conf().user_conf().op_type_name() == "identity_buffer");
  int64_t regst_num = user_op::UserOpConfWrapper(sole_op->op_conf()).attr<int64_t>("buffer_size");
  CHECK_GT(regst_num, 0);
  std::shared_ptr<RegstDesc> regst = ProduceRegst("out", false, regst_num, regst_num);
  ForEachOutDataEdge([&](TaskEdge* edge) { edge->AddRegst("out", regst); });
}

void BufferCompTaskNode::ConsumeAllRegsts() {
  ConsumeRegst("in", SoleInDataEdge()->GetSoleRegst());
}

void BufferCompTaskNode::BuildExecGphAndRegst() {
  std::shared_ptr<RegstDesc> in_regst = GetSoleConsumedRegst("in");
  std::shared_ptr<RegstDesc> out_regst = GetProducedRegst("out");
  ExecNode* exec_node = mut_exec_gph().NewNode();
  exec_node->mut_op() = op();
  exec_node->BindBnWithRegst(op()->SoleIbn(), in_regst);
  out_regst->AddLbi(op()->BnInOp2Lbi(op()->SoleObn()));
  exec_node->BindBnWithRegst(op()->SoleObn(), out_regst);
  exec_node->InferBlobDescs(parallel_ctx());
}

#ifdef WITH_CUDA
REGISTER_NAMED_TASK_STREAM_INDEX_GETTER(DeviceType::kCUDA, TaskType::kBuffer, "PIPELINE_BUFFER")
#endif
REGISTER_NAMED_TASK_STREAM_INDEX_GETTER(DeviceType::kCPU, TaskType::kBuffer, "PIPELINE_BUFFER")

REGISTER_USER_OP_COMP_TASK_NODE_TYPE("identity_buffer", BufferCompTaskNode);

}  // namespace oneflow
