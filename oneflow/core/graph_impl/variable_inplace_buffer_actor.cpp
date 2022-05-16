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

namespace oneflow {

class VariableInplaceBufferCompTaskNode final : public CompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(VariableInplaceBufferCompTaskNode);
  VariableInplaceBufferCompTaskNode() = default;
  ~VariableInplaceBufferCompTaskNode() override = default;

  void ProduceAllRegstsAndBindEdges() override;
  void ConsumeAllRegsts() override;

  TaskType GetTaskType() const override { return TaskType::kVariableInplaceBuffer; }

 private:
  void BuildExecGphAndRegst() override;
};

void VariableInplaceBufferCompTaskNode::ConsumeAllRegsts() {
  ConsumeRegst("in", SoleInDataEdge()->GetSoleRegst());
}

void VariableInplaceBufferCompTaskNode::ProduceAllRegstsAndBindEdges() {
  // NOTE(chengcheng):
  //   In compiler: VariableInplaceBufferCompTaskNode has 1 regst num for inplace.
  //   In Runtime: VariableInplaceBufferActor will create n regst by buffer size with same dptr.
  std::shared_ptr<RegstDesc> out_regst = ProduceRegst("out", false, 1, 1);
  ForEachOutDataEdge([&](TaskEdge* edge) { edge->AddRegst("out", out_regst); });
}

void VariableInplaceBufferCompTaskNode::BuildExecGphAndRegst() {
  ExecNode* node = mut_exec_gph().NewNode();
  std::shared_ptr<const Operator> sole_op = op();
  node->mut_op() = sole_op;
  std::shared_ptr<RegstDesc> in_regst = GetSoleConsumedRegst("in");
  std::shared_ptr<RegstDesc> out_regst = GetProducedRegst("out");
  in_regst->set_enable_reuse_mem(false);
  node->BindBnWithRegst(sole_op->SoleIbn(), in_regst);
  out_regst->AddLbi(sole_op->BnInOp2Lbi(sole_op->SoleObn()));
  // NOTE(chengcheng):
  //   Using force inplace make sure that buffer inplace succeed.
  out_regst->set_force_inplace_consumed_regst_desc_id(in_regst->regst_desc_id());
  node->BindBnWithRegst(sole_op->SoleObn(), out_regst);
  node->InferBlobDescs(parallel_ctx());
}

REGISTER_COMP_TASK_STREAM_INDEX_GETTER(TaskType::kVariableInplaceBuffer);

REGISTER_USER_OP_COMP_TASK_NODE_TYPE("_variable_inplace_buffer", VariableInplaceBufferCompTaskNode);

}  // namespace oneflow
