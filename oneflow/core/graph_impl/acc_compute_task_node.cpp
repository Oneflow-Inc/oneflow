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
#include "oneflow/core/framework/framework.h"

namespace oneflow {

class AccCompTaskNode final : public CompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AccCompTaskNode);
  AccCompTaskNode() = default;
  ~AccCompTaskNode() = default;
  TaskType GetTaskType() const override { return TaskType::kAcc; }
  void BuildExecGphAndRegst() override;
  void ProduceAllRegstsAndBindEdges() override;
  void ConsumeAllRegsts() override;
};

void AccCompTaskNode::ProduceAllRegstsAndBindEdges() {
  std::shared_ptr<RegstDesc> regst = ProduceRegst("out", false);
  ForEachOutDataEdge([&](TaskEdge* edge) { edge->AddRegst("out", regst); });
}

void AccCompTaskNode::ConsumeAllRegsts() { ConsumeRegst("in", SoleInDataEdge()->GetSoleRegst()); }

void AccCompTaskNode::BuildExecGphAndRegst() {
  std::shared_ptr<RegstDesc> in_regst = GetSoleConsumedRegst("in");
  std::shared_ptr<RegstDesc> out_regst = GetProducedRegst("out");
  ExecNode* exec_node = mut_exec_gph().NewNode();
  exec_node->mut_op() = op();
  exec_node->BindBnWithRegst(op()->SoleIbn(), in_regst);
  out_regst->AddLbi(op()->BnInOp2Lbi(op()->SoleObn()));
  exec_node->BindBnWithRegst(op()->SoleObn(), out_regst);
  exec_node->InferBlobDescs(parallel_ctx());
  out_regst->ForEachLbi([out_regst](const LogicalBlobId& lbi) {
    const BlobDesc* blob_desc = out_regst->GetBlobDesc(lbi);
    CHECK_EQ(blob_desc->is_dynamic(), false);
  });
}

REGISTER_USER_OP_COMP_TASK_NODE_TYPE("acc", AccCompTaskNode);

}  // namespace oneflow
