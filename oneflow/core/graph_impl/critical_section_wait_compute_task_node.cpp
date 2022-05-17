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

class CriticalSectionWaitTickCompTaskNode final : public CompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CriticalSectionWaitTickCompTaskNode);
  CriticalSectionWaitTickCompTaskNode() = default;
  ~CriticalSectionWaitTickCompTaskNode() = default;

  bool IsMeaningLess() override { return false; }
  TaskType GetTaskType() const override { return TaskType::kCriticalSectionWaitTick; }

 private:
  void ProduceAllRegstsAndBindEdges() override;
  void ConsumeAllRegsts() override;
  void BuildExecGphAndRegst() override;
};

void CriticalSectionWaitTickCompTaskNode::ProduceAllRegstsAndBindEdges() {
  ProduceRegst("out", false, 128, 128);
  ForEachOutDataEdge([&](TaskEdge* edge) { BindEdgeWithProducedRegst(edge, "out"); });
}

void CriticalSectionWaitTickCompTaskNode::ConsumeAllRegsts() {
  ConsumeRegst("in");
  ForEachInDataEdge([&](TaskEdge* edge) { ConsumeRegst("in", edge->GetSoleRegst()); });
}

void CriticalSectionWaitTickCompTaskNode::BuildExecGphAndRegst() {
  ExecNode* node = mut_exec_gph().NewNode();
  node->mut_op() = op();
  const std::list<std::shared_ptr<RegstDesc>>& in_regsts = GetConsumedRegst("in");
  for (const std::string& ibn : node->op()->input_bns()) {
    node->BindBnWithOneOfTheRegsts(ibn, in_regsts);
  }
  std::shared_ptr<RegstDesc> out_regst = GetProducedRegst("out");
  for (const std::string& obn : node->op()->output_bns()) {
    const LogicalBlobId& lbi = node->op()->BnInOp2Lbi(obn);
    out_regst->AddLbi(lbi);
    node->BindBnWithRegst(obn, out_regst);
  }
  node->InferBlobDescs(parallel_ctx());
}

REGISTER_INDEPENDENT_TASK_STREAM_INDEX_GETTER(TaskType::kCriticalSectionWaitTick);

REGISTER_SYSTEM_OP_COMP_TASK_NODE_TYPE(OperatorConf::kCriticalSectionWaitTickConf,
                                       CriticalSectionWaitTickCompTaskNode);

}  // namespace oneflow
