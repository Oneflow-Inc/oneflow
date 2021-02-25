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
#include "oneflow/core/graph/src_subset_tick_compute_task_node.h"
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

void SrcSubsetTickCompTaskNode::ProduceAllRegstsAndBindEdges() {
  ProduceRegst("out", false, 1, 1);
  ForEachOutDataEdge([&](TaskEdge* edge) { BindEdgeWithProducedRegst(edge, "out"); });
}

void SrcSubsetTickCompTaskNode::ConsumeAllRegsts() {
  ConsumeRegst("in");
  ForEachInDataEdge([&](TaskEdge* edge) { ConsumeRegst("in", edge->GetSoleRegst()); });
}

void SrcSubsetTickCompTaskNode::BuildExecGphAndRegst() {
  ExecNode* node = mut_exec_gph().NewNode();
  node->mut_op() = logical_node()->SoleOp();
  const std::list<std::shared_ptr<RegstDesc>>& in_regsts = GetConsumedRegst("in");
  for (const std::string& ibn : node->op()->input_bns()) {
    node->TryBindBnWithOneOfTheRegsts(ibn, in_regsts);
  }
  std::shared_ptr<RegstDesc> out_regst = GetProducedRegst("out");
  for (const std::string& obn : node->op()->output_bns()) {
    const LogicalBlobId& lbi = node->op()->BnInOp2Lbi(obn);
    out_regst->AddLbi(lbi);
    node->BindBnWithRegst(obn, out_regst);
  }
  node->InferBlobDescs(parallel_ctx());
}

void SrcSubsetTickCompTaskNode::InferProducedDataRegstTimeShape() {
  auto time_shape = (*in_edges().begin())->src_node()->GetFastestInputOutputTimeShape();
  for (TaskEdge* edge : in_edges()) {
    CHECK(time_shape->elem_cnt() == edge->src_node()->GetFastestInputOutputTimeShape()->elem_cnt());
  }
  ForEachProducedDataRegst([time_shape](const std::string& name, RegstDesc* regst) {
    *regst->mut_data_regst_time_shape() = time_shape;
  });
}

REGISTER_TICK_TOCK_TASK_TYPE(TaskType::kSrcSubsetTick);

}  // namespace oneflow
