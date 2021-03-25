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
#include "oneflow/core/graph/task_graph.h"
#include "oneflow/core/operator/variable_op.h"

namespace oneflow {

class DistributeSplitCompTaskNode final : public CompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DistributeSplitCompTaskNode);
  DistributeSplitCompTaskNode() = default;
  ~DistributeSplitCompTaskNode() = default;

  void ProduceAllRegstsAndBindEdges() override;
  void ConsumeAllRegsts() override;

  TaskType GetTaskType() const override { return TaskType::kDistributeSplit; }

 private:
  void BuildExecGphAndRegst() override;
  void BuildExecGphStructAndBindInRegst();
  void BuildOutRegst();
};

void DistributeSplitCompTaskNode::ProduceAllRegstsAndBindEdges() {
  ProduceRegst("out", true);
  ForEachOutDataEdge([&](TaskEdge* edge) { BindEdgeWithProducedRegst(edge, "out"); });
}

void DistributeSplitCompTaskNode::ConsumeAllRegsts() {
  ForEachInDataEdge([&](TaskEdge* edge) { ConsumeRegst("in", edge->GetSoleRegst()); });
}

void DistributeSplitCompTaskNode::BuildExecGphAndRegst() {
  BuildExecGphStructAndBindInRegst();
  BuildOutRegst();
  mut_exec_gph().TopoForEachNode([this](ExecNode* node) { node->InferBlobDescs(parallel_ctx()); });
}

void DistributeSplitCompTaskNode::BuildExecGphStructAndBindInRegst() {
  ExecNode* cur_node = mut_exec_gph().NewNode();
  cur_node->mut_op() = this->op();
  for (const std::string& ibn : cur_node->op()->input_bns()) {
    cur_node->BindBnWithRegst(ibn, GetSoleConsumedRegst("in"));
  }
}

void DistributeSplitCompTaskNode::BuildOutRegst() {
  std::shared_ptr<RegstDesc> out_regst = GetProducedRegst("out");
  mut_exec_gph().ForEachNode([&](ExecNode* cur_node) {
    const auto& obn = cur_node->op()->output_bns().Get(parallel_ctx()->parallel_id());
    out_regst->AddLbi(cur_node->op()->BnInOp2Lbi(obn));
    cur_node->BindBnWithRegst(obn, out_regst);
  });
  // NOTE: we can ONLY set inplace when regst has ONLY ONE blob
  auto in_regst = GetSoleConsumedRegst("in");
  if (in_regst->NumOfLbi() == 1) {
    out_regst->set_hint_inplace_consumed_regst_desc_id(in_regst->regst_desc_id());
  }
}

REGISTER_SYSTEM_OP_COMP_TASK_NODE_TYPE(OperatorConf::kDistributeSplitConf,
                                       DistributeSplitCompTaskNode);

REGISTER_SYSTEM_OP_COMP_TASK_NODE_TYPE(OperatorConf::kDistributeCloneConf,
                                       DistributeSplitCompTaskNode);

}  // namespace oneflow
