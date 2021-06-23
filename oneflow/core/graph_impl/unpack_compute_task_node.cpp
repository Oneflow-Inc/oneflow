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

namespace oneflow {

class UnpackCompTaskNode final : public CompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(UnpackCompTaskNode);
  UnpackCompTaskNode() = default;
  ~UnpackCompTaskNode() override = default;

  TaskType GetTaskType() const override { return TaskType::kUnpack; }

  void ProduceAllRegstsAndBindEdges() override;
  void ConsumeAllRegsts() override;

 private:
  void BuildExecGphAndRegst() override;
};

void UnpackCompTaskNode::ProduceAllRegstsAndBindEdges() {
  ProduceRegst("out", false);
  ForEachOutDataEdge([&](TaskEdge* edge) { BindEdgeWithProducedRegst(edge, "out"); });
}

void UnpackCompTaskNode::ConsumeAllRegsts() {
  ConsumeRegst("in", SoleInDataEdge()->GetSoleRegst());
}

void UnpackCompTaskNode::BuildExecGphAndRegst() {
  ExecNode* exec_node = mut_exec_gph().NewNode();
  exec_node->mut_op() = op();
  exec_node->BindBnWithRegst(op()->SoleIbn(), GetSoleConsumedRegst("in"));

  std::shared_ptr<RegstDesc> out_regst = GetProducedRegst("out");
  out_regst->AddLbi(op()->BnInOp2Lbi(op()->SoleObn()));
  exec_node->BindBnWithRegst(op()->SoleObn(), out_regst);
  exec_node->InferBlobDescs(parallel_ctx());
}

REGISTER_USER_OP_COMP_TASK_NODE_TYPE("unpack", UnpackCompTaskNode);

}  // namespace oneflow
