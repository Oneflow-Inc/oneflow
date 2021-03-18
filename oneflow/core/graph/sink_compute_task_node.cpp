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
#include "oneflow/core/graph/sink_compute_task_node.h"

namespace oneflow {

void SinkCompTaskNode::ProduceAllRegstsAndBindEdges() {}

void SinkCompTaskNode::ConsumeAllRegsts() {
  ForEachInDataEdge([&](TaskEdge* edge) { ConsumeRegst("in", edge->GetSoleRegst()); });
}

void SinkCompTaskNode::BuildExecGphAndRegst() {
  ExecNode* node = mut_exec_gph().NewNode();
  node->mut_op() = shared_op();
  for (const std::string& ibn : node->op()->input_bns()) {
    node->BindBnWithOneOfTheRegsts(ibn, GetConsumedRegst("in"));
  }
  CHECK(node->op()->tmp_bns().empty());
  CHECK(node->op()->output_bns().empty());
}

}  // namespace oneflow
