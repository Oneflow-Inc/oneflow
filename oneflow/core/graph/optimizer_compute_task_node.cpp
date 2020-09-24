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
#include "oneflow/core/graph/logical_node.h"
#include "oneflow/core/graph/optimizer_compute_task_node.h"

namespace oneflow {

void OptimizerCompTaskNode::ConsumeAllRegsts() {
  ForEachInDataEdge([&](TaskEdge* edge) {
    for (const auto& regst : edge->GetRegsts()) { ConsumeRegst("in", regst); }
  });
}

void OptimizerCompTaskNode::ProduceAllRegstsAndBindEdges() { ProduceRegst("tmp", false, 1, 1); }

void OptimizerCompTaskNode::BuildExecGphAndRegst() {
  ExecNode* node = mut_exec_gph().NewNode();
  std::shared_ptr<Operator> sole_op = this->logical_node()->SoleOp();
  node->mut_op() = sole_op;
  const std::list<std::shared_ptr<RegstDesc>>& in_regsts = GetConsumedRegst("in");
  for (const auto& ibn : node->op()->input_bns()) {
    node->BindBnWithOneOfTheRegsts(ibn, in_regsts);
  }
  node->AddBnToRegstAndBindIt(&Operator::tmp_bns, GetProducedRegst("tmp"));
  node->InferBlobDescs(parallel_ctx());
}

void OptimizerCompTaskNode::InferProducedDataRegstTimeShape() {
  ForEachProducedDataRegst([](const std::string& name, RegstDesc* regst) {
    regst->mut_data_regst_time_shape()->reset(
        new Shape({GlobalJobDesc().TotalBatchNum(), static_cast<int64_t>(1)}));
  });
}

}  // namespace oneflow
